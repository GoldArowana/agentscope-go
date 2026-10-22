package formatter

import (
	"github.com/agentscope-ai/agentscope-go/v2/pkg/agentscope/message"
)

// OpenAIFormatter formats messages for OpenAI-compatible APIs (OpenAI, DashScope, DeepSeek).
type OpenAIFormatter struct {
	// audioAsDataURL wraps base64 audio in a "data:;base64," URL (DashScope).
	audioAsDataURL bool
	// SupportsThinking enables reasoning_content field (DashScope/DeepSeek).
	SupportsThinking bool
	// SupportedInputMediaTypes lists glob patterns of accepted media types (e.g. "image/*").
	SupportedInputMediaTypes []string
	// ToolNameInResult includes the tool name in tool result messages (Ollama requires this).
	ToolNameInResult bool
}

// Format expands each internal message into ordered provider messages. A merged
// agent reply can contain assistant tool calls, tool results and further replies.
func (f *OpenAIFormatter) Format(msgs []*message.Msg) ([]map[string]any, error) {
	var result []map[string]any
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		blocks := msg.GetContentBlocks()
		var segments [][]message.ContentBlock
		start := 0
		for i, block := range blocks {
			if _, ok := block.(message.ToolResultBlock); !ok {
				continue
			}
			if start < i {
				segments = append(segments, blocks[start:i])
			}
			segments = append(segments, blocks[i:i+1])
			start = i + 1
		}
		if start < len(blocks) || len(blocks) == 0 {
			segments = append(segments, blocks[start:])
		}
		for _, segment := range segments {
			// Reuse the provider's block conversion without changing stored history.
			part := *msg
			part.Content = segment
			formatted, err := f.formatMsg(&part)
			if err != nil {
				return nil, err
			}
			if formatted != nil {
				result = append(result, formatted)
			}
		}
	}
	return result, nil
}

func (f *OpenAIFormatter) formatMsg(msg *message.Msg) (map[string]any, error) {
	m := map[string]any{"role": string(msg.Role)}

	blocks := msg.GetContentBlocks()
	if len(blocks) == 0 {
		m["content"] = ""
		return m, nil
	}

	var textParts []string
	var thinkingParts []string
	var toolCalls []map[string]any
	var toolResultID, toolResultContent, toolResultName string
	var multimodalParts []map[string]any
	isToolResult := false
	hasMultimodal := false

	for _, b := range blocks {
		switch blk := b.(type) {
		case message.TextBlock:
			textParts = append(textParts, blk.Text)
		case message.ThinkingBlock:
			thinkingParts = append(thinkingParts, blk.Thinking)
		case message.ToolCallBlock:
			tc := map[string]any{
				"id":   blk.ID,
				"type": "function",
				"function": map[string]any{
					"name":      blk.Name,
					"arguments": blk.Input,
				},
			}
			toolCalls = append(toolCalls, tc)
		case message.ToolResultBlock:
			isToolResult = true
			toolResultID = blk.ID
			toolResultContent = ConvertToolResultToString(blk.Output)
			toolResultName = blk.Name
		case message.HintBlock:
			textParts = append(textParts, blk.GetHintText())
		case message.DataBlock:
			// Use the checked internal variant so invalid audio subtypes
			// surface as Format errors instead of being sent to the API
			// (upstream #2301).
			formatted, err := formatDataBlock(blk, f.SupportedInputMediaTypes, f.audioAsDataURL)
			if err != nil {
				return nil, err
			}
			if formatted != nil {
				multimodalParts = append(multimodalParts, formatted)
				hasMultimodal = true
			}
		}
	}

	if isToolResult {
		result := map[string]any{
			"role":         "tool",
			"tool_call_id": toolResultID,
			"content":      toolResultContent,
		}
		if f.ToolNameInResult {
			result["tool_name"] = toolResultName
		}
		return result, nil
	}

	if len(toolCalls) > 0 {
		m["tool_calls"] = toolCalls
	}

	textContent := joinStrings(textParts)

	if hasMultimodal {
		var contentArray []map[string]any
		if textContent != "" {
			contentArray = append(contentArray, map[string]any{"type": "text", "text": textContent})
		}
		contentArray = append(contentArray, multimodalParts...)
		m["content"] = contentArray
	} else if textContent != "" {
		m["content"] = textContent
	} else if len(toolCalls) > 0 {
		m["content"] = nil
	} else {
		m["content"] = ""
	}

	if f.SupportsThinking && len(thinkingParts) > 0 {
		m["reasoning_content"] = joinStrings(thinkingParts)
	}

	return m, nil
}

func joinStrings(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0]
	}
	result := parts[0]
	for _, p := range parts[1:] {
		result += p
	}
	return result
}

// --- DashScope Formatter (extends OpenAI with video/audio/thinking support) ---

// DashScopeFormatter formats messages for DashScope APIs.
// Extends OpenAI with video_url, input_audio, and reasoning_content support.
type DashScopeFormatter struct {
	OpenAIFormatter
}

// NewDashScopeFormatter creates a formatter for DashScope.
func NewDashScopeFormatter() *DashScopeFormatter {
	return &DashScopeFormatter{
		OpenAIFormatter: OpenAIFormatter{
			audioAsDataURL:           true,
			SupportsThinking:         true,
			SupportedInputMediaTypes: []string{"image/*", "audio/*", "video/*"},
		},
	}
}

// FormatMultiAgent formats messages for multi-agent conversations.
func (f *OpenAIFormatter) FormatMultiAgent(msgs []*message.Msg, currentAgent string) ([]map[string]any, error) {
	return formatMultiAgentOpenAI(f, msgs, currentAgent)
}

// NewOpenAIFormatter creates a formatter for standard OpenAI APIs.
func NewOpenAIFormatter() *OpenAIFormatter {
	return &OpenAIFormatter{
		SupportsThinking:         false,
		SupportedInputMediaTypes: []string{"image/*", "audio/*"},
	}
}

// FormatMultiAgent formats messages for multi-agent DashScope conversations.
func (f *DashScopeFormatter) FormatMultiAgent(msgs []*message.Msg, currentAgent string) ([]map[string]any, error) {
	return formatMultiAgentOpenAI(f, msgs, currentAgent)
}

// --- Anthropic Formatter ---

// AnthropicFormatter formats messages for Anthropic's Messages API.
// Supports image content blocks and ThinkingBlock with signature.
type AnthropicFormatter struct {
	SupportedInputMediaTypes []string
}

func (f *AnthropicFormatter) Format(msgs []*message.Msg) ([]map[string]any, error) {
	var result []map[string]any
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		if msg.Role == message.RoleSystem {
			continue
		}
		result = append(result, f.expandMsg(msg)...)
	}
	return result, nil
}

// expandMsg formats one internal message as ordered wire messages. Agents may
// merge a whole tool round into a single assistant message (thinking, text,
// tool_use and tool_result blocks interleaved), but the Anthropic Messages
// API only accepts tool_result blocks in USER messages — an assistant message
// carrying them is wire-invalid (strict endpoints reject it; lenient proxies
// accept it but the model never sees the results). The expansion walks the
// blocks in order, flushing whenever the wire role changes, so a merged
// round becomes [assistant(...tool_use), user(tool_result...)] pairs with
// every result after its call.
func (f *AnthropicFormatter) expandMsg(msg *message.Msg) []map[string]any {
	if msg == nil {
		return nil
	}
	role := string(msg.Role)
	if role == "tool" {
		role = "user"
	}

	blocks := msg.GetContentBlocks()
	if len(blocks) == 0 {
		return nil
	}

	var (
		out      []map[string]any
		curRole  string
		content  []map[string]any
	)
	flush := func() {
		if len(content) == 0 {
			return
		}
		out = append(out, map[string]any{"role": curRole, "content": content})
		content = nil
	}
	appendBlock := func(targetRole string, block map[string]any) {
		if block == nil {
			return
		}
		if curRole == "" {
			curRole = targetRole
		}
		if targetRole != curRole {
			flush()
			curRole = targetRole
		}
		content = append(content, block)
	}

	for _, b := range blocks {
		// tool_result blocks always belong to a user message; everything
		// else keeps the message's own (possibly remapped) role.
		_, isToolResult := b.(message.ToolResultBlock)
		blockRole := role
		if isToolResult {
			blockRole = "user"
		}
		for _, formatted := range f.formatBlock(b) {
			appendBlock(blockRole, formatted)
		}
	}
	flush()
	return out
}

// formatBlock renders one content block in the Anthropic wire format.
func (f *AnthropicFormatter) formatBlock(b message.ContentBlock) []map[string]any {
	switch blk := b.(type) {
	case message.TextBlock:
		if blk.Text == "" {
			return nil
		}
		return []map[string]any{{"type": "text", "text": blk.Text}}
	case message.ThinkingBlock:
		if blk.Thinking == "" {
			return nil
		}
		tb := map[string]any{
			"type":     "thinking",
			"thinking": blk.Thinking,
		}
		if sig, ok := blk.Extra["signature"]; ok && sig != "" {
			tb["signature"] = sig
		}
		return []map[string]any{tb}
	case message.RedactedThinkingBlock:
		rtb := map[string]any{
			"type": "redacted_thinking",
		}
		if blk.Data != "" {
			rtb["data"] = blk.Data
		}
		return []map[string]any{rtb}
	case message.ToolCallBlock:
		inputObj := jsonLoadsWithRepair(blk.Input)
		return []map[string]any{{
			"type":  "tool_use",
			"id":    blk.ID,
			"name":  blk.Name,
			"input": inputObj,
		}}
	case message.ToolResultBlock:
		resultText := ConvertToolResultToString(blk.Output)
		if resultText == "" {
			resultText = "(empty tool output)"
		}
		return []map[string]any{{
			"type":        "tool_result",
			"tool_use_id": blk.ID,
			"content":     resultText,
		}}
	case message.HintBlock:
		var out []map[string]any
		for _, sub := range hintContent(blk) {
			switch h := sub.(type) {
			case message.TextBlock:
				out = append(out, map[string]any{"type": "text", "text": h.Text})
			case message.DataBlock:
				if formatted := f.formatAnthropicDataBlock(h); formatted != nil {
					out = append(out, formatted)
				}
			}
		}
		return out
	case message.DataBlock:
		if formatted := f.formatAnthropicDataBlock(blk); formatted != nil {
			return []map[string]any{formatted}
		}
		return nil
	}
	return nil
}

func (f *AnthropicFormatter) formatAnthropicDataBlock(blk message.DataBlock) map[string]any {
	mt := blk.GetMediaType()
	if !SupportsMediaType(f.SupportedInputMediaTypes, mt) {
		return nil
	}
	switch src := blk.Source.(type) {
	case message.Base64Source:
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": src.MediaType,
				"data":       src.Data,
			},
		}
	case message.URLSource:
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type": "url",
				"url":  src.URL,
			},
		}
	}
	return nil
}

// FormatMultiAgent formats messages for multi-agent Anthropic conversations.
// Uses <history> tags for conversation context from other agents.
func (f *AnthropicFormatter) FormatMultiAgent(msgs []*message.Msg, currentAgent string) ([]map[string]any, error) {
	var formatted []map[string]any
	for _, msg := range msgs {
		if msg == nil || msg.Role == message.RoleSystem {
			continue
		}
		expanded := f.expandMsg(msg)
		if len(expanded) == 0 {
			continue
		}
		if msg.Name != "" && msg.Name != currentAgent {
			// Prepend the sender name to the first wire message of the
			// expansion (the message's own role).
			if m := expanded[0]; m != nil {
				if role, _ := m["role"].(string); role == "user" || role == "assistant" {
					if content, ok := m["content"].([]map[string]any); ok && len(content) > 0 {
						nameBlock := map[string]any{"type": "text", "text": "[" + msg.Name + "]:"}
						m["content"] = append([]map[string]any{nameBlock}, content...)
					}
				}
			}
		}
		formatted = append(formatted, expanded...)
	}

	return formatted, nil
}

// NewAnthropicFormatter creates a formatter for Anthropic's Messages API.
func NewAnthropicFormatter() *AnthropicFormatter {
	return &AnthropicFormatter{
		SupportedInputMediaTypes: []string{"image/*"},
	}
}

// ExtractSystemPrompt extracts the system message text from a message list.
// Anthropic requires system prompt to be passed separately, not in the messages array.
func ExtractSystemPrompt(msgs []*message.Msg) string {
	for _, msg := range msgs {
		if msg != nil && msg.Role == message.RoleSystem {
			if txt := msg.GetTextContent("\n"); txt != nil {
				return *txt
			}
		}
	}
	return ""
}
