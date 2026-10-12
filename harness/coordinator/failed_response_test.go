package coordinator

import (
	"context"
	"errors"
	"strings"
	"testing"
	"testing/synctest"

	"github.com/unreallabsai/unreal-agent/harness/contextbuilder"
	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/sessionstore"
)

// TestFailedModelResponseFailsRun covers a provider answering HTTP 200 with a
// response.failed event: the adapter fills Response.Failure without a Go error,
// the coordinator must fail the run, and the failed response's tool calls must
// not be scheduled. Compaction turns keep retrying (see
// TestAutomaticCompactionRejectedResponseRetries).
func TestFailedModelResponseFailsRun(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		run := newStopTestRun(t, 0)
		run.start(t)
		run.input(t, externalEvent(t, 0, "pending", "Continue work"))
		response := llm.Response{
			Failure: &llm.Failure{Code: "insufficient_quota", Message: "quota exhausted"},
			Output: []llm.Item{{Type: llm.ItemToolCall, Data: llm.ToolCall{
				CallID: "call-failed", Name: "view_image", Arguments: `{}`,
			}}},
		}
		run.respond(t, 0, response)
		select {
		case err := <-run.done:
			if err == nil || !strings.Contains(err.Error(), "insufficient_quota") {
				t.Fatalf("Run error = %v, want a failure naming insufficient_quota", err)
			}
		default:
			t.Fatal("Run continued after a failed model response")
		}
		if len(run.current.state.toolCalls) != 0 {
			t.Fatalf("failed response carried tool calls into local state: %+v", run.current.state.toolCalls)
		}
		var recorded bool
		for _, response := range run.store.appendedResponses {
			recorded = recorded || response.Response.Failure != nil
		}
		if !recorded {
			t.Fatal("failed response was not preserved in the session")
		}
	})
}

// TestFailedModelResponseToolCallsNotRestoredOnResume covers the same failure
// replayed from the session store: the recorded tool calls must not run after a
// session resume.
func TestFailedModelResponseToolCallsNotRestoredOnResume(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		store, registry := independentToolCalls(t, 0)
		response := sessionstore.ModelResponse{
			TurnID: "turn-1",
			Response: llm.Response{
				Failure: &llm.Failure{Code: "insufficient_quota", Message: "quota exhausted"},
				Output: []llm.Item{{Type: llm.ItemToolCall, Data: llm.ToolCall{
					CallID: "call-failed", Name: "view_image", Arguments: `{}`,
				}}},
			},
		}
		store.items = append(store.items,
			storedItem(sessionstore.Sequence(len(store.items)+1), sessionstore.ItemModelResponse, response),
		)
		adapter := &fakeAdapter{respond: func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{}, errors.New("restore must not request a model")
		}}
		current := newTestCoordinatorWithAdapter(
			store, newTestInbox(t), newFakeOperationManager(), contextbuilder.NewBuilder(), registry, adapter,
		)
		if err := current.restore(t.Context()); err != nil {
			t.Fatal(err)
		}
		if len(current.state.toolCalls) != 0 {
			t.Fatalf("resume restored tool calls from a failed response: %+v", current.state.toolCalls)
		}
	})
}
