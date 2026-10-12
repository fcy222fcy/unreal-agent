package agentrunner

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/unreallabsai/unreal-agent/cmd/internal/providers"
	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/sessionstore"
)

// failedQuotaResponse mimics a provider answering HTTP 200 with a
// response.failed event: the adapter fills Response.Failure without a Go error
// and without a stop reason, because the response never completed.
func failedQuotaResponse() llm.Response {
	return llm.Response{
		ID:      "response-failed",
		Failure: &llm.Failure{Code: "insufficient_quota", Message: "quota exhausted"},
	}
}

// failedQuotaToolCallResponse is failedQuotaResponse with a tool call in the
// failed output, as a provider that streamed a call before failing would return.
func failedQuotaToolCallResponse() llm.Response {
	response := failedQuotaResponse()
	response.Output = []llm.Item{{Type: llm.ItemToolCall, Data: llm.ToolCall{
		CallID: "call-failed", Name: "Bash", Arguments: `{"command":"printf resumed"}`,
	}}}
	return response
}

func finishedResponse() llm.Response {
	return llm.Response{
		ID: "response-finished", Stop: llm.StopComplete,
		Output: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleAssistant, Text: "finished"}}},
	}
}

func TestRunMainExitsNonzeroAfterFailedModelResponse(t *testing.T) {
	t.Setenv(providers.APIKeyEnvironment, "secret")
	client := &fakeClient{}
	client.respond = func(context.Context, llm.Request) (llm.Response, error) {
		return failedQuotaResponse(), nil
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	var stdout, stderr bytes.Buffer
	code := RunMain(
		ctx,
		[]string{"-workspace", t.TempDir(), "-session-directory", t.TempDir()},
		func(name string) string {
			if name == providers.APIKeyEnvironment {
				return "secret"
			}
			return ""
		},
		func() []string { return nil },
		strings.NewReader(`{"prompt":"hello"}`),
		&stdout,
		&stderr,
		testConfig(client),
	)
	if code != 1 {
		t.Fatalf("exit = %d, want 1; stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"Code":"insufficient_quota"`) {
		t.Fatalf("failed response was not recorded in the session output: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "insufficient_quota") {
		t.Fatalf("stderr = %q, want the failure code", stderr.String())
	}
}

func TestRunMainFailedResponseToolCallsDoNotRunAfterResume(t *testing.T) {
	t.Setenv(providers.APIKeyEnvironment, "secret")
	workspace, sessions := t.TempDir(), t.TempDir()
	request := fmt.Sprintf(`{"session_id":"failed-resume","messages":[{"role":"user","content":"run it","message_id":%q}]}`, "69621f8d-4f4d-49a5-8f7d-3b24fd855c01")
	resume := fmt.Sprintf(`{"session_id":"failed-resume","messages":[{"role":"user","content":"again","message_id":%q}]}`, "7dd7cc9c-0446-4dd9-9d84-6f26e2f89260")
	client := &fakeClient{}
	client.respond = func(context.Context, llm.Request) (llm.Response, error) {
		client.mu.Lock()
		defer client.mu.Unlock()
		client.calls++
		if client.calls == 1 {
			return failedQuotaToolCallResponse(), nil
		}
		return finishedResponse(), nil
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	run := func(request string, output io.Writer) int {
		return RunMain(
			ctx,
			[]string{"-workspace", workspace, "-session-directory", sessions},
			func(name string) string {
				if name == providers.APIKeyEnvironment {
					return "secret"
				}
				if name == "SHELL" {
					return "/bin/sh"
				}
				return ""
			},
			func() []string { return []string{"PATH=/usr/bin:/bin", "UNREAL_HARNESS_LLM_API_KEY=secret"} },
			strings.NewReader(request),
			output,
			io.Discard,
			testConfig(client),
		)
	}
	var first bytes.Buffer
	if code := run(request, &first); code != 1 {
		t.Fatalf("failed run exit = %d, want 1", code)
	}
	if !strings.Contains(first.String(), `"Code":"insufficient_quota"`) {
		t.Fatalf("failed run did not record the failure: %s", first.String())
	}
	client.mu.Lock()
	callsAfterFailure := client.calls
	client.mu.Unlock()
	var second bytes.Buffer
	if code := run(resume, &second); code != 0 {
		t.Fatalf("resumed run exit = %d, want 0", code)
	}
	for _, kind := range itemKinds(t, second.String()) {
		if kind == sessionstore.ItemToolCallStatus {
			t.Fatal("resumed run executed the failed response's tool call")
		}
	}
	client.mu.Lock()
	defer client.mu.Unlock()
	if got := client.calls - callsAfterFailure; got != 1 {
		t.Fatalf("resumed run made %d model requests, want 1", got)
	}
}
