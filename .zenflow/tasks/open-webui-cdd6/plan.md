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
<!-- chat-id: f88e43b4-91a7-4a02-aef2-b0adb1e42274 -->

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

### [x] Step: Add Open WebUI to deployment profiles
<!-- chat-id: aa795d94-e91c-4472-9b84-8f541230597b -->

Add the `open-webui` service with `profiles: ["ui"]` to all three deployment profiles, plus env vars and volume.

- Add `open-webui` service to `deployments/urandom/docker-compose.yml`
- Add `open-webui-data` named volume to `deployments/urandom/docker-compose.yml`
- Add `OPEN_WEBUI_PORT` and `OPEN_WEBUI_AUTH` to `deployments/urandom/.env.example`
- Repeat for `deployments/firefly-1/`
- Repeat for `deployments/_template/`
- Verify: `docker compose --profile ui config` passes in each profile directory (syntax check)
- Verify: `docker compose config` (no profile) does NOT include `open-webui` service

---

### [x] Step: Create qr-sampler Filter Function for Open WebUI
<!-- chat-id: 164157f9-f9d8-4275-82b1-12a9869c3d72 -->

Write the Open WebUI Filter Function that injects `qr_*` per-request parameters into vLLM requests via Valves.

- Create `examples/open-webui/qr_sampler_filter.py` with:
  - `Pipeline` class with `type = "filter"`
  - `Valves` inner class exposing per-request fields from `_PER_REQUEST_FIELDS` in `config.py`
  - `inlet()` method that injects `qr_*` keys as top-level body fields
  - `outlet()` passthrough
  - Proper docstrings and metadata header (title, author, version, description)
- Create `examples/open-webui/qr_sampler_filter.json` — Open WebUI importable JSON containing the filter source
- Verify: JSON is valid and contains the complete Python source
- Verify: Valve field names match the `qr_*` keys that `resolve_config()` accepts (cross-reference `_PER_REQUEST_FIELDS`)

---

### [x] Step: Update documentation
<!-- chat-id: 741bf761-a443-4e71-97ed-4725a466fc90 -->

Update all READMEs to document the Open WebUI integration and filter function.

- Add "Web UI (optional)" section to `deployments/urandom/README.md`
- Add "Web UI (optional)" section to `deployments/firefly-1/README.md`
- Add "Web UI (optional)" section to `deployments/_template/README.md`
- Update `deployments/README.md` quick start to mention `--profile ui`
- Create `examples/open-webui/README.md` with filter function docs (import steps, Valves config, architecture)
- Add prominent "Web UI" section to main `README.md` recommending Open WebUI
- Write report to `{@artifacts_path}/report.md`

