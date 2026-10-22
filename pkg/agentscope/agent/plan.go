package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/event"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/message"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/model"
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/tool"
)

// WithPlanTool registers the update_plan tool (codex parity): the model
// maintains a visible step plan for multi-step work. The plan lives in the
// checkpointed agent state — it survives compression and crash-resume — and
// is injected into the model input after the summary on every call, so the
// model never loses track after a compaction. Each successful update emits a
// plan_updated custom event for hosts to render.
func WithPlanTool() AgentOption {
	return func(a *UnifiedAgent) { a.planToolEnabled = true }
}

// WithContextRemainingTool registers the get_context_remaining tool (codex
// parity): the model can check its remaining budget before starting an
// expensive subtask. The estimate is the max of the byte heuristic and the
// provider-reported input tokens of the last call, subtracted from the
// resolved context window (ContextConfig.ContextSize, else the model card /
// default).
func WithContextRemainingTool() AgentOption {
	return func(a *UnifiedAgent) { a.contextRemainingEnabled = true }
}

// updatePlanFromTool is the UpdatePlanFunc wired into the update_plan tool.
// Validation already happened in the tool; this only records the plan.
func (a *UnifiedAgent) updatePlanFromTool(_ context.Context, _ string, plan []tool.PlanItem) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Plan = append([]tool.PlanItem(nil), plan...)
	return nil
}

// contextRemainingForTool is the ContextRemainingFunc wired into the
// get_context_remaining tool.
func (a *UnifiedAgent) contextRemainingForTool(ctx context.Context) (int64, bool) {
	ctxSize := 0
	if a.contextCfg != nil {
		ctxSize = a.contextCfg.ContextSize
	}
	if ctxSize == 0 {
		// Mirror the compression path: the fallback model's smaller window
		// governs, since a retried call must fit it.
		ctxSize = model.ResolveContextSize(a.model, defaultContextSize)
		if a.modelCfg.FallbackModel != nil {
			ctxSize = min(ctxSize, model.ResolveContextSize(a.modelCfg.FallbackModel, defaultContextSize))
		}
	}
	msgs := a.prepareModelInput(ctx)
	est := a.model.CountTokens(msgs, a.toolkit.GetToolSchemas())
	a.mu.Lock()
	if a.lastModelInputTokens > est {
		est = a.lastModelInputTokens
	}
	a.mu.Unlock()
	left := int64(ctxSize) - int64(est)
	if left < 0 {
		left = 0
	}
	return left, true
}

// planPromptMsg builds the user message that carries the current plan into
// the model input (and into the compression split accounting).
func planPromptMsg(agentName string, plan []tool.PlanItem) *message.Msg {
	return message.UserMsg(agentName, "[Current plan]:\n"+formatPlanForPrompt(plan))
}

// formatPlanForPrompt renders the plan for model-input injection.
func formatPlanForPrompt(plan []tool.PlanItem) string {
	var sb strings.Builder
	for i, item := range plan {
		fmt.Fprintf(&sb, "%d. [%s] %s\n", i+1, item.Status, item.Step)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// planEventValue converts a plan-update metadata payload into the
// plan_updated custom event value.
func planEventValue(pm tool.PlanMetadata) map[string]any {
	items := make([]any, len(pm.Plan))
	for i, it := range pm.Plan {
		items[i] = map[string]any{"step": it.Step, "status": it.Status}
	}
	v := map[string]any{"plan": items}
	if pm.Explanation != "" {
		v["explanation"] = pm.Explanation
	}
	return v
}

// emitPlanUpdateIfPresent turns a successful update_plan response's metadata
// into a plan_updated custom event. update_plan is not concurrency-safe, so
// this only ever runs on the sequential path with a live channel.
func emitPlanUpdateIfPresent(ctx context.Context, ch chan<- event.Event, replyID string, state message.ToolResultState, md map[string]any) {
	if state != message.ToolResultSuccess || md == nil {
		return
	}
	pm, ok := md[tool.PlanMetadataKey].(tool.PlanMetadata)
	if !ok {
		return
	}
	emit(ctx, ch, event.NewCustomEvent(replyID, "plan_updated", planEventValue(pm)))
}
