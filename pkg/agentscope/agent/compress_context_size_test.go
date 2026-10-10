package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/message"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/middleware"
)

// These fixtures exercise the real compression chain with an offline model;
// they establish threshold and history handling, not summary quality.
func TestCompressContext_ContextSizeResolution(t *testing.T) {
	for _, tc := range []struct {
		name           string
		override       int
		modelWindow    int
		tokens         int
		summaryFails   bool
		wantCompressed bool
	}{
		{"zero_below_model_window", 0, 1000, 10, false, false},
		{"negative_one_below_model_window", -1, 1000, 10, false, false},
		{"negative_below_model_window_failed_summary", -100, 1000, 10, true, false},
		{"zero_below_default_window", 0, 0, 1000, true, false},
		{"negative_below_default_window", -100, 0, 1000, true, false},
		{"positive_override_below_threshold", 2000, 1000, 900, true, false},
		{"positive_override_above_threshold", 100, 1000, 90, false, true},
		{"zero_above_model_window", 0, 1000, 900, false, true},
		{"negative_above_model_window", -100, 1000, 900, false, true},
		{"zero_above_model_window_failed_summary", 0, 1000, 900, true, true},
		{"negative_above_model_window_failed_summary", -100, 1000, 900, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mock := &compressionMockModel{contextSize: tc.modelWindow, tokenCount: tc.tokens}
			if tc.summaryFails {
				mock.chatErr = errors.New("offline summary failure")
			}
			cfg := &ContextConfig{ContextSize: tc.override}
			a := NewUnifiedAgent("test", "prompt", mock, WithContextConfig(cfg))
			a.state.Summary = "previous summary"
			history := []*message.Msg{
				message.UserMsg("user", "first"),
				message.AssistantMsg("bot", "second"),
				message.UserMsg("user", "third"),
			}
			a.state.Context = history
			if err := a.compressContext(context.Background()); err != nil {
				t.Fatal(err)
			}
			calls := atomic.LoadInt64(&mock.chatCallCount)
			t.Logf("override=%d modelWindow=%d tokens=%d summaryCalls=%d retained=%d", tc.override, tc.modelWindow, tc.tokens, calls, len(a.state.Context))
			if cfg.ContextSize != tc.override {
				t.Fatal("caller-owned config was mutated")
			}
			if !tc.wantCompressed {
				if calls != 0 {
					t.Errorf("summary calls = %d, want 0 below resolved threshold", calls)
				}
				if a.state.Summary != "previous summary" || len(a.state.Context) != len(history) {
					t.Fatalf("below-threshold history changed: summary=%q retained=%d", a.state.Summary, len(a.state.Context))
				}
				for i := range history {
					if a.state.Context[i] != history[i] {
						t.Errorf("history message %d replaced", i)
					}
				}
				return
			}
			if calls == 0 {
				t.Error("above-threshold compression did not call summary model")
			}
			if len(a.state.Context) != 1 {
				t.Errorf("compression should retain newest message, retained=%d", len(a.state.Context))
			} else if text := a.state.Context[0].GetTextContent(""); text == nil || *text != "third" {
				t.Error("compression did not retain newest message content")
			}
			if tc.summaryFails {
				if a.state.Summary != "previous summary" {
					t.Errorf("failed summary replaced previous summary: %q", a.state.Summary)
				}
			} else if a.state.Summary == "previous summary" || a.state.Summary == "" {
				t.Error("successful compression did not replace summary")
			}
		})
	}
}

func TestCompressContext_ContextSizeAcrossEntryPoints(t *testing.T) {
	for _, mode := range []string{"middleware", "tool"} {
		for _, tokens := range []int{10, 600} {
			t.Run(mode+"_"+fmt.Sprint(tokens), func(t *testing.T) {
				mock := &compressionMockModel{contextSize: 1000, tokenCount: tokens}
				a := NewUnifiedAgent("test", "prompt", mock,
					WithContextConfig(&ContextConfig{ContextSize: -1}),
					WithMiddlewares(&middleware.BaseMiddleware{MiddlewareKey: "passthrough"}),
					WithAgentDrivenCompression(),
				)
				a.state.Context = []*message.Msg{message.UserMsg("user", "history")}
				wantCompressed := mode == "tool" && tokens == 600
				if mode == "tool" {
					response, err := a.toolkit.CallTool(context.Background(), "compress_context", map[string]any{})
					if err != nil {
						t.Fatal(err)
					}
					text, ok := response.Content[0].(message.TextBlock)
					if !ok || strings.Contains(text.Text, "Context compressed:") != wantCompressed {
						t.Errorf("unexpected tool compression report: %+v", response)
					}
				} else if err := a.compressContext(context.Background()); err != nil {
					t.Fatal(err)
				}
				calls := atomic.LoadInt64(&mock.chatCallCount)
				if (calls > 0) != wantCompressed {
					t.Errorf("summary calls = %d, wantCompressed=%v", calls, wantCompressed)
				}
				if !wantCompressed && (len(a.state.Context) != 1 || a.state.Summary != "") {
					t.Error("below-threshold entry point changed history")
				}
			})
		}
	}
}
