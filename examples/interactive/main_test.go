package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRunApprovesPlanAndResumesForBackgroundReport(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader("print commands\napprove\n"), &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	want := []string{
		"Question: Which kind of help should the demo produce?",
		"Question answer: print commands",
		"Approved plan: Prepare an offline command report",
		"Parked reply: Background worker launched; waiting for its report.",
		"Resumed after injected completion.",
		"Agent status: completed",
		"Final background report: go test ./... and go vet ./...",
	}
	for _, text := range want {
		if !strings.Contains(out.String(), text) {
			t.Errorf("output missing %q:\n%s", text, out.String())
		}
	}
}

func TestRunRevisesPlanFromFeedbackWithoutRequestCounting(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader("print commands\nfeedback: keep one concise step\napprove\n"), &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(out.String(), "Revised plan: Prepare one concise offline command report") {
		t.Fatalf("revised proposal not observed:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "Approved plan: Prepare one concise offline command report") {
		t.Fatalf("revised proposal was not approved:\n%s", out.String())
	}
}

func TestRunExecutesEditedPlan(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader("print commands\nedit:Edited local checks | Record both Go checks\n"), &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(out.String(), "Approved plan: Edited local checks") {
		t.Fatalf("edited plan was not approved:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "Plan step executed: run local Go checks") {
		t.Fatalf("edited plan was not executed:\n%s", out.String())
	}
}

func TestRunRejectsPlanGracefully(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader("print commands\nreject\n"), &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(out.String(), "Plan rejected; no work was executed.") {
		t.Fatalf("rejection not reported:\n%s", out.String())
	}
	if strings.Contains(out.String(), "Parked reply:") {
		t.Fatalf("background stage ran after rejection:\n%s", out.String())
	}
}

func TestRunReturnsEOFWhileQuestionIsPending(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader(""), &out)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("run error = %v, want EOF", err)
	}
}

func TestRunHonorsCancellationBeforeReading(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var out bytes.Buffer
	err := run(ctx, strings.NewReader("print commands\napprove\n"), &out)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("run error = %v, want context cancellation", err)
	}
}

type promptBuffer struct {
	mu       sync.Mutex
	buf      bytes.Buffer
	prompted chan struct{}
	once     sync.Once
}

func (w *promptBuffer) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n, err := w.buf.Write(p)
	if strings.Contains(w.buf.String(), "Question:") {
		w.once.Do(func() { close(w.prompted) })
	}
	return n, err
}

func TestRunCancellationReleasesPendingQuestion(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader, writer := io.Pipe()
	out := &promptBuffer{prompted: make(chan struct{})}
	done := make(chan error, 1)
	go func() { done <- run(ctx, reader, out) }()

	select {
	case <-out.prompted:
	case <-time.After(2 * time.Second):
		t.Fatal("question prompt was not displayed")
	}
	cancel()
	_ = writer.Close()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("run error = %v, want context cancellation", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation did not release the pending question")
	}
}
