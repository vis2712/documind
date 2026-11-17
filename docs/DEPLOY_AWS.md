# Deploying DocuMind to AWS

Two options. **Option 1 is what I recommend for a portfolio demo**: one server, about $15/month, up in under an hour.
Option 2 is the production-shaped architecture; it's good to be able to explain it in interviews even if you
only run Option 1.

---

## Option 1: Single EC2 instance with Docker Compose (recommended for the demo)

```
Internet ──► EC2 (t3.small, Ubuntu)
              └─ docker compose: ui:8501 · api:8000 · worker · postgres+pgvector · redis
```

1. **Launch the instance** (EC2 console → Launch instance)
   - Ubuntu 24.04, `t3.small` (2 GB RAM; the reranker and OCR need more than 1 GB), 20 GB gp3 disk
   - Security group: allow SSH (22) from *My IP* only, and 8501 (UI) from anywhere
   - Create and download a key pair, e.g. `documind.pem`
2. **Install Docker**
   ```bash
   ssh -i documind.pem ubuntu@<public-ip>
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker ubuntu && exit   # log in again so the group applies
   ```
3. **Deploy**
   ```bash
   ssh -i documind.pem ubuntu@<public-ip>
   git clone https://github.com/<you>/documind.git && cd documind
   cp .env.example .env && nano .env        # LLM_API_KEY, JWT_SECRET
   docker compose up -d --build
   docker compose ps                        # all services "running"/"healthy"
   ```
4. Open `http://<public-ip>:8501`.
5. **Update after a push**: `git pull && docker compose up -d --build`

Hardening worth doing before you share the link: put Caddy in front for HTTPS with a free domain
(`caddy reverse-proxy --from your.domain --to localhost:8501`), keep ports 5433/6379/8000 closed in the
security group (they are only needed inside Docker), and set a monthly AWS budget alert.

---

## Option 2: Managed services (production architecture)

```
               ┌──────────── ALB (HTTPS, ACM certificate) ────────────┐
Internet ──►   │  /        → ECS Fargate service: ui                   │
               │  /api/*   → ECS Fargate service: api (2+ tasks)       │
               └───────────────────────────────────────────────────────┘
                     api ──► ElastiCache Redis (RQ queue) ──► ECS service: worker (autoscale on queue depth)
                     api, worker ──► RDS PostgreSQL 17 (pgvector extension)
                     api, worker ──► S3 bucket (uploaded files), via the ECS task IAM role
                     secrets (LLM key, JWT secret, DB password) ──► AWS Secrets Manager → task env
```

| Piece | AWS service | Notes |
|---|---|---|
| Images | ECR | `documind-api` (also used by the worker) and `documind-ui` |
| API, worker, UI | ECS Fargate | Same image for API and worker; worker command `python -m app.worker` |
| Database | RDS PostgreSQL 17 | Run `CREATE EXTENSION vector;` (the app does this on startup) |
| Queue | ElastiCache Redis | `REDIS_URL=redis://<endpoint>:6379/0` |
| Files | S3 | `STORAGE_BACKEND=s3`, `S3_BUCKET=...`; task role needs `s3:GetObject/PutObject/DeleteObject` |
| Secrets | Secrets Manager | Injected as env vars in the task definition |
| Logs | CloudWatch Logs | `awslogs` log driver; the API logs latency/tokens/cost per answer |
| CI/CD | GitHub Actions | On push to main: test → build → push to ECR → `aws ecs update-service --force-new-deployment` |

Push images to ECR:
```bash
aws ecr create-repository --repository-name documind-api
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.ap-south-1.amazonaws.com
docker build --platform linux/amd64 -t <acct>.dkr.ecr.ap-south-1.amazonaws.com/documind-api:latest .
docker push <acct>.dkr.ecr.ap-south-1.amazonaws.com/documind-api:latest
```
(`--platform linux/amd64` matters when building on an Apple Silicon Mac for x86 Fargate.)

Rough monthly cost at demo scale: RDS db.t4g.micro ~$15, ElastiCache t4g.micro ~$12, ALB ~$18, Fargate 3 small
tasks ~$30. That's ~$75/month, which is why Option 1 is better for a demo.

### Interview talking points
- **Why a separate worker?** OCR and embedding take seconds to minutes; the API returns 202 immediately and
  scales independently of the workers. Jobs survive API restarts and are retried with backoff.
- **Why S3, not local disk?** Fargate tasks are ephemeral and the API and worker run on different machines.
- **Scaling retrieval:** pgvector HNSW is fine to roughly tens of millions of chunks; every query is filtered
  by `owner_id` (tenant isolation). Beyond that: partition by tenant, or move vectors to a dedicated store.
