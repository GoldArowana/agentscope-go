package agent

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/event"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/message"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/model"
)

// precompressMockModel counts tokens per message so the estimate grows with
// the context, and serves structured-summary responses to compression calls.
type precompressMockModel struct {
	contextSize      int
	perMessageTokens int64
	chatCalls        atomic.Int64
	chatErr          error
}

func (m *precompressMockModel) Chat(_ context.Context, _ []*message.Msg, _ ...model.CallOption) (*model.ChatResponse, error) {
	m.chatCalls.Add(1)
	if m.chatErr != nil {
		return nil, m.chatErr
	}
	return summaryToolCallResponse(&model.ChatUsage{InputTokens: 11, OutputTokens: 7}), nil
}

func (m *precompressMockModel) ChatStream(_ context.Context, _ []*message.Msg, _ ...model.CallOption) (<-chan model.ChatResponse, error) {
	return nil, model.ErrStreamNotSupported
}

func (m *precompressMockModel) CountTokens(msgs []*message.Msg, _ []model.ToolSchema) int {
	return int(atomic.LoadInt64(&m.perMessageTokens)) * len(msgs)
}

func (m *precompressMockModel) ContextSize() int { return m.contextSize }

func (m *precompressMockModel) ModelName() string { return "precompress-mock" }

func newPrecompressAgent(mock *precompressMockModel, cfg *ContextConfig) *UnifiedAgent {
	return NewUnifiedAgent("t", "sys", mock, WithContextConfig(cfg))
}

func seedContext(a *UnifiedAgent, n int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := 0; i < n; i++ {
		a.state.Context = append(a.state.Context, message.NewMsg("u", message.RoleUser, "message body"))
	}
}

// waitPrecompress polls until the background generation settles: either a
// pending result is installed, or the run finished without one (failure or
// nothing to compress).
func waitPrecompress(t *testing.T, a *UnifiedAgent) *precompressResult {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		p, running := a.pendingPrecompress, a.precompressRunning
		a.mu.Unlock()
		if !running {
			return p
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("background pre-compression did not settle")
	return nil
}

// TestPrecompress_ArmsUnderThreshold: between the precompress ratio and the
// hard trigger, compressContext kicks off background generation without
// touching the live state.
func TestPrecompress_ArmsUnderThreshold(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, ReserveRatio: 0.3, PrecompressRatio: 0.5,
	})
	// system + 4 context messages = 500 tokens: past the 500 arm line, under
	// the 800 hard trigger.
	seedContext(a, 4)

	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	p := waitPrecompress(t, a)
	if p == nil {
		t.Fatal("expected a pending pre-compression result")
	}
	if got := mock.chatCalls.Load(); got != 1 {
		t.Errorf("chat calls = %d, want 1 (the background generation)", got)
	}
	a.mu.Lock()
	summary, ctxLen := a.state.Summary, len(a.state.Context)
	a.mu.Unlock()
	if summary != "" || ctxLen != 4 {
		t.Errorf("arming must not mutate state: summary=%q ctxLen=%d", summary, ctxLen)
	}
	// The generation burned tokens; usage is recorded for the reply loop to
	// drain.
	u := a.takeCompressionUsage()
	if u == nil || u.InputTokens != 11 || u.OutputTokens != 7 {
		t.Errorf("compression usage = %+v, want in:11 out:7", u)
	}
}

// TestPrecompress_AppliesAtHardTrigger: with a valid result pending, the hard
// trigger installs it without a new model call; messages appended after the
// snapshot survive raw.
func TestPrecompress_AppliesAtHardTrigger(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, ReserveRatio: 0.3, PrecompressRatio: 0.5,
	})
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	if waitPrecompress(t, a) == nil {
		t.Fatal("expected a pending pre-compression result")
	}
	a.mu.Lock()
	boundary := a.state.Context[3]
	a.mu.Unlock()

	// Conversation continues past the hard trigger: 8 context messages +
	// system = 900 tokens >= 800.
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext at trigger: %v", err)
	}

	if got := mock.chatCalls.Load(); got != 1 {
		t.Errorf("chat calls = %d, want 1: the ready result must apply without regenerating", got)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.pendingPrecompress != nil {
		t.Error("pending result must be consumed by the apply")
	}
	if !strings.Contains(a.state.Summary, "T") {
		t.Errorf("summary = %q, want the generated summary", a.state.Summary)
	}
	// Reserve budget 300 fits one message next to the system prompt, so the
	// snapshot reserved its last message; the four appended afterwards stay.
	if len(a.state.Context) != 5 {
		t.Fatalf("context len = %d, want 5 (1 reserved + 4 appended)", len(a.state.Context))
	}
	if a.state.Context[0] != boundary {
		t.Error("reserved message must be the live boundary message, not a snapshot copy")
	}
}

func TestPrecompress_BelowRatioDoesNotArm(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, PrecompressRatio: 0.5,
	})
	seedContext(a, 2) // 300 tokens total: below the 500 arm line

	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	a.mu.Lock()
	running, pending := a.precompressRunning, a.pendingPrecompress
	a.mu.Unlock()
	if running || pending != nil {
		t.Errorf("must not arm below the ratio: running=%v pending=%v", running, pending)
	}
	if got := mock.chatCalls.Load(); got != 0 {
		t.Errorf("chat calls = %d, want 0", got)
	}
}

func TestPrecompress_DisabledByDefault(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, // PrecompressRatio unset
	})
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	a.mu.Lock()
	running := a.precompressRunning
	a.mu.Unlock()
	if running {
		t.Error("precompression must be opt-in")
	}
	if got := mock.chatCalls.Load(); got != 0 {
		t.Errorf("chat calls = %d, want 0", got)
	}
}

func TestPrecompress_RatioClampedWhenNotBelowTrigger(t *testing.T) {
	cfg := (&ContextConfig{TriggerRatio: 0.8, PrecompressRatio: 0.85}).withDefaults()
	if cfg.PrecompressRatio != 0 {
		t.Errorf("PrecompressRatio >= TriggerRatio must disable: got %v", cfg.PrecompressRatio)
	}
	cfg = (&ContextConfig{PrecompressRatio: -1}).withDefaults()
	if cfg.PrecompressRatio != 0 {
		t.Errorf("negative PrecompressRatio must normalize to 0, got %v", cfg.PrecompressRatio)
	}
	cfg = (&ContextConfig{TriggerRatio: 0.8, PrecompressRatio: 0.5}).withDefaults()
	if cfg.PrecompressRatio != 0.5 {
		t.Errorf("valid PrecompressRatio changed: %v", cfg.PrecompressRatio)
	}
}

// TestPrecompress_StaleResultDiscarded: a result whose base summary no longer
// matches (another compaction landed meanwhile) is discarded, and the hard
// trigger falls back to synchronous compression.
func TestPrecompress_StaleResultDiscarded(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, ReserveRatio: 0.3, PrecompressRatio: 0.5,
	})
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	if waitPrecompress(t, a) == nil {
		t.Fatal("expected a pending pre-compression result")
	}

	// Simulate an interleaved compaction (checkpoint rollback has the same
	// shape): the base summary the generation built upon is gone.
	a.mu.Lock()
	a.state.Summary = "someone else compacted"
	a.mu.Unlock()

	seedContext(a, 4) // 8 context + system = 900 >= 800
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext at trigger: %v", err)
	}
	if got := mock.chatCalls.Load(); got != 2 {
		t.Errorf("chat calls = %d, want 2: stale result must fall back to a synchronous call", got)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.pendingPrecompress != nil {
		t.Error("stale result must be discarded")
	}
	if !strings.Contains(a.state.Summary, "T") {
		t.Errorf("summary = %q, want the synchronously generated one", a.state.Summary)
	}
}

// TestPrecompress_GenerationFailureDiscards: a failed background generation
// leaves no pending result and does not wedge the agent — the synchronous
// path still compresses at the hard trigger.
func TestPrecompress_GenerationFailureDiscards(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100, chatErr: errors.New("boom")}
	a := newPrecompressAgent(mock, &ContextConfig{
		ContextSize: 1000, TriggerRatio: 0.8, ReserveRatio: 0.3, PrecompressRatio: 0.5,
	})
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	if p := waitPrecompress(t, a); p != nil {
		t.Fatalf("failed generation must leave no pending result, got %+v", p)
	}
	if got := mock.chatCalls.Load(); got != 1 {
		t.Errorf("chat calls = %d, want 1", got)
	}

	mock.chatErr = nil
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("synchronous fallback: %v", err)
	}
	a.mu.Lock()
	summary := a.state.Summary
	a.mu.Unlock()
	if !strings.Contains(summary, "T") {
		t.Errorf("summary = %q, want the synchronous summary after background failure", summary)
	}
}

// TestPrecompress_NotArmedWithContentBlockCompaction: the block-level
// splitter has different granularity, so precompression stays off under it.
func TestPrecompress_NotArmedWithContentBlockCompaction(t *testing.T) {
	mock := &precompressMockModel{contextSize: 1000, perMessageTokens: 100}
	a := NewUnifiedAgent("t", "sys", mock,
		WithContextConfig(&ContextConfig{
			ContextSize: 1000, TriggerRatio: 0.8, PrecompressRatio: 0.5,
		}),
		WithContentBlockCompaction(),
	)
	seedContext(a, 4)
	if err := a.compressContext(context.Background()); err != nil {
		t.Fatalf("compressContext: %v", err)
	}
	a.mu.Lock()
	running := a.precompressRunning
	a.mu.Unlock()
	if running {
		t.Error("precompression must not arm under content-block compaction")
	}
}

// TestEmitTurnStats_DrainsCompressionUsage: compression usage that lands
// after the last Reason-state drain (e.g. a background pre-compression
// finishing during the final model call) is accounted to the ending reply:
// a model-call-end event first, then turn_stats carrying the tokens.
func TestEmitTurnStats_DrainsCompressionUsage(t *testing.T) {
	mock := &compressionMockModel{tokenCount: 1, contextSize: 100}
	a := NewUnifiedAgent("t", "sys", mock)
	a.beginTurnStats()
	a.recordCompressionUsage(&model.ChatUsage{InputTokens: 11, OutputTokens: 7})

	ch := make(chan event.Event, 4)
	a.emitTurnStats(context.Background(), ch, "rid")
	close(ch)

	var kinds []event.EventType
	var stats *event.CustomEvent
	var callEnd *event.ModelCallEndEvent
	for evt := range ch {
		kinds = append(kinds, evt.GetEventType())
		switch e := evt.(type) {
		case event.CustomEvent:
			if e.Name == "turn_stats" {
				stats = &e
			}
		case event.ModelCallEndEvent:
			callEnd = &e
		}
	}
	if len(kinds) != 2 || kinds[0] != event.EventModelCallEnd || kinds[1] != event.EventCustom {
		t.Fatalf("event sequence = %v, want [model_call_end custom]", kinds)
	}
	if callEnd == nil || callEnd.InputTokens != 11 || callEnd.OutputTokens != 7 {
		t.Errorf("model call end = %+v, want in:11 out:7", callEnd)
	}
	if stats == nil {
		t.Fatal("missing turn_stats")
	}
	if stats.Value["model_calls"] != 1 || stats.Value["input_tokens"] != 11 || stats.Value["output_tokens"] != 7 {
		t.Errorf("turn_stats = %+v, want the drained usage included", stats.Value)
	}
	if a.takeCompressionUsage() != nil {
		t.Error("drain must clear the accumulator")
	}
}
