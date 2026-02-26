# Spec and build

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Agent Instructions

Ask the user questions when anything is unclear or needs their input. This includes:
- Ambiguous or incomplete requirements
- Technical decisions that affect architecture or user experience
- Trade-offs that require business context

Do not make assumptions on important decisions — get clarification first.

If you are blocked and need user clarification, mark the current step with `[!]` in plan.md before stopping.

---

## Workflow Steps

### [x] Step: Technical Specification
<!-- chat-id: 953ae008-10de-4aa3-bf5e-5634a15bc61f -->

Assess the task's difficulty, as underestimating it leads to poor outcomes.
- easy: Straightforward implementation, trivial bug fix or feature
- medium: Moderate complexity, some edge cases or caveats to consider
- hard: Complex logic, many caveats, architectural considerations, or high-risk changes

Create a technical specification for the task that is appropriate for the complexity level:
- Review the existing codebase architecture and identify reusable components.
- Define the implementation approach based on established patterns in the project.
- Identify all source code files that will be created or modified.
- Define any necessary data model, API, or interface changes.
- Describe verification steps using the project's test and lint commands.

Save the output to `{@artifacts_path}/spec.md` with:
- Technical context (language, dependencies)
- Implementation approach
- Source code structure changes
- Data model / API / interface changes
- Verification approach

If the task is complex enough, create a detailed implementation plan based on `{@artifacts_path}/spec.md`:
- Break down the work into concrete tasks (incrementable, testable milestones)
- Each task should reference relevant contracts and include verification steps
- Replace the Implementation step below with the planned tasks

Rule of thumb for step size: each step should represent a coherent unit of work (e.g., implement a component, add an API endpoint, write tests for a module). Avoid steps that are too granular (single function).

Important: unit tests must be part of each implementation task, not separate tasks. Each task should implement the code and its tests together, if relevant.

Save to `{@artifacts_path}/plan.md`. If the feature is trivial and doesn't warrant this breakdown, keep the Implementation step below as is.

---

### [x] Step: Update Dockerfile.vllm defaults and create per-source Docker Compose files
<!-- chat-id: f05dbea6-39c0-4b58-9c96-4067e5f727ab -->

Update the shared vLLM Dockerfile with new model defaults, then create self-contained `docker-compose.yml` files for each deployment profile. Delete the old centralized compose and override files.

- Update `examples/docker/Dockerfile.vllm`: change `HF_MODEL` default to `Qwen/Qwen2.5-1.5B-Instruct`, add `--dtype half --gpu-memory-utilization 0.90` to `CMD`
- Delete `examples/docker/docker-compose.yml` (replaced by per-source compose files)
- Delete `deployments/urandom/docker-compose.override.yml` (replaced by self-contained compose)
- Create `deployments/urandom/docker-compose.yml` — self-contained with both vllm + entropy-server services
- Create `deployments/urandom/.env.example` — annotated env template
- Create `deployments/firefly-1/docker-compose.yml` — self-contained with vllm service only (firefly-1 is external)
- Create `deployments/firefly-1/.env.example` — annotated env template (sanitized, no real API key)
- Create `deployments/_template/docker-compose.yml` — annotated compose template
- Create `deployments/_template/.env.example` — annotated env template
- Verify: run `docker compose -f deployments/urandom/docker-compose.yml config` (and similar for each profile)

### [x] Step: Update deployment profile READMEs
<!-- chat-id: 7c7a3cfa-c30d-4a03-9573-5e4677c1b8bb -->

Rewrite each deployment profile's README to be a self-contained setup guide with the new simplified workflow.

- Update `deployments/README.md` — overview of new per-source pattern, link to each profile
- Update `deployments/_template/README.md` — simplified: "cp .env.example .env, edit, docker compose up"
- Update `deployments/urandom/README.md` — self-contained 3-step quickstart
- Update `deployments/firefly-1/README.md` — self-contained 3-step quickstart

### [x] Step: Restructure main README
<!-- chat-id: 878f0d02-edd4-4829-aec2-4cd865a67d63 -->

Rewrite the main README.md to prioritize external entropy sources, deprioritize system entropy, update all model references, and simplify Docker instructions.

- Update all `meta-llama/Llama-3.2-1B` references to `Qwen/Qwen2.5-1.5B-Instruct`
- Restructure Quick start: lead with external entropy Docker setup, make system entropy a secondary/fallback option
- Simplify Docker commands: `cd deployments/urandom && docker compose up --build` instead of complex multi-file paths
- Update curl examples with new model name
- Update "Deployment profiles" section to reflect new self-contained pattern
- Verify: grep for any remaining `Llama-3.2` references, check all markdown links

### [x] Step: Final verification and report
<!-- chat-id: b2e9c7a5-9104-44c2-acb7-154486bd62d6 -->

Run all verification checks and write the implementation report.

- Grep entire repo for `meta-llama/Llama-3.2-1B` — should find zero matches
- Verify all new docker-compose.yml files are valid YAML
- Verify no broken file references in READMEs
- Run `ruff check` on any modified Python files (if any)
- Write report to `{@artifacts_path}/report.md`
