@productize @productize/plan.md @productize/activity.md

We are finding an optimal PyTorch training setup for pcdiff model (just this one, voxelzation model is already trained) in this repo. We will be using `uv` for virtual environment + installing python packages (already in @.venv).

First read @productize/activity.md to see what was recently accomplished.

Familiarize yourself with @productize/PRD.md.

Open @productize/plan.md and choose the single highest priority task where passes is false.

Work on exactly ONE task: implement the change.

After implementing, run multi-GPU pcdiff training run with early stopping mentioned in @productize/PRD.md and @productize/plan.md.

Append a dated progress entry to @productize/activity.md describing what you changed and link to logs / wandb logs.

Update that task's passes in @productize/plan.md from false to true.

Make one git commit for that task only with a clear message.

Do not git init, do not change remotes.

ONLY WORK ON A SINGLE TASK.

When ALL tasks have passes true, output <promise>COMPLETE</promise>