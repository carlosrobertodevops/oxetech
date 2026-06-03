# OxeTech Academy Monorepo - Agent Instructions

This repository contains multiple educational courses, labs, and assignments for the OxeTech Academy. It is structured as a monorepo with distinct sub-projects.

## General Navigation
Before making changes, **always look for and read the `CLAUDE.md` or `README.md` file within the specific sub-directory you are working on.** They contain critical instructions, commands, and constraints tailored to that specific project.

## Project Boundaries & Specific Workflows

### 1. `iaam-tarde/` (AI & Machine Learning)
- **Files:** Jupyter Notebooks (`.ipynb`) and their Jupytext Python mirrors (`.py`).
- **Workflow:** **Never edit `.ipynb` files directly.** Edit the `.py` files instead, which use the `percent` format (`# %%`).
- **Sync:** Sync changes back to notebooks using `jupytext --sync arquivo.py`.
- **Solutions:** In `iaam-tarde/av2/`, solutions are injected into student templates using the `patch.py` script via regex replacement.

### 2. `ccii-noite/desafio-aula07/` (Terraform & AWS)
- **Structure:** Contains completely separated and independent `aws/` (real AWS) and `localstack/` (local emulator) directories. They do not share modules.
- **Commands:** 
  - For AWS: `terraform apply -var-file="envs/production.tfvars"`
  - For LocalStack: Use `tflocal` (e.g., `tflocal plan`). It auto-loads `localstack.auto.tfvars`.
- **Constraint:** Native `hashicorp/aws` resources only (pinned `~> 6.0`). Strict naming conventions apply (read its `CLAUDE.md`).

### 3. `ccii-noite/0x3-cl0ud-l4bs/` (Cloud Labs Platform)
- **Stack:** A monolithic FastAPI server (`labs/lab_server.py`) running on `localhost:8765`. Frontend is a single React SPA transpilated in-browser via Babel standalone (no build step, no npm).
- **Workflow:** Use the `Makefile`.
  - `make setup-labs` (install deps)
  - `make lab` (run server)
  - `make localstack-up` (start LocalStack via Docker)

### 4. `dwfs-tarde/` (Frontend & Fullstack Web)
- Contains various web development exercises and projects using HTML, CSS, JavaScript, and Node.js.
- Standard web development workflows apply.

## Agent Constraints
- Do not run `terraform apply` or `destroy` in the `ccii-noite` projects unless explicitly requested by the user. Deliverables are typically code-only.
- Respect the subagent constraints defined in the sub-project `CLAUDE.md` files (e.g., `ccii-noite/desafio-aula07/CLAUDE.md` limits subagent types due to prompt size).