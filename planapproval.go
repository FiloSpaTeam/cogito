package cogito

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/mudler/cogito/structures"
)

// PlanDecision describes how an automatically generated plan should proceed.
// Approved accepts the proposed plan, or Plan when it is non-nil. A rejected
// decision with non-empty Feedback asks Cogito to revise the proposal.
type PlanDecision struct {
	Approved bool
	Plan     *structures.Plan
	Feedback string
}

// ErrPlanRejected is returned when automatic plan approval is declined without
// feedback or the configured number of feedback revisions has been exhausted.
var ErrPlanRejected = errors.New("plan rejected by user")

// WithPlanApproval installs a synchronous approval callback for plans produced
// by automatic planning. The callback receives the execution context and must
// return when that context is canceled. A nil callback disables approval.
func WithPlanApproval(fn func(context.Context, *structures.Plan, *structures.Goal) PlanDecision) Option {
	return func(o *Options) {
		o.planApproval = fn
	}
}

func approveAutomaticPlan(
	llm LLM,
	f Fragment,
	plan *structures.Plan,
	goal *structures.Goal,
	o *Options,
	opts ...Option,
) (*structures.Plan, error) {
	planningContext := f
	planningContext.Messages = slices.Clone(f.Messages)

	for revisions := 0; ; revisions++ {
		if err := o.context.Err(); err != nil {
			return nil, err
		}
		decision := o.planApproval(o.context, plan, goal)
		if err := o.context.Err(); err != nil {
			return nil, err
		}
		if decision.Approved {
			if decision.Plan != nil {
				plan = decision.Plan
			}
			if len(plan.Subtasks) == 0 {
				return nil, fmt.Errorf("approved plan has no subtasks")
			}
			return plan, nil
		}

		feedback := strings.TrimSpace(decision.Feedback)
		if feedback == "" || revisions >= o.maxAdjustmentAttempts {
			return nil, ErrPlanRejected
		}

		proposal := fmt.Sprintf("Proposed plan: %s\nSubtasks:\n%s", plan.Description, strings.Join(plan.Subtasks, "\n"))
		planningContext = planningContext.AddMessage(AssistantMessageRole, proposal)
		planningContext = planningContext.AddMessage(UserMessageRole, feedback)
		if err := o.context.Err(); err != nil {
			return nil, err
		}
		var err error
		plan, err = ExtractPlan(llm, planningContext, goal, opts...)
		if err != nil {
			return nil, fmt.Errorf("failed to revise plan: %w", err)
		}
	}
}
