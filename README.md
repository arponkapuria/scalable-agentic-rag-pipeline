## Full-Scale Deployment

This branch contains the high-availability, production-scale deployment architecture of the Scalable Agentic RAG Pipeline.

> A Work in Progress branch !

### Overview

The `dev/aws-full-scale` branch represents the enterprise-scale deployment designed for high throughput, fault tolerance, and distributed inference workloads. Compared to the cost-optimized deployment in `dev/cost-optimized`, this architecture prioritizes scalability, availability, and operational resilience.

### Key Characteristics

- Multi-node Amazon EKS deployment
- Ray-based distributed ingestion and inference
- Dedicated GPU inference workloads powered by vLLM
- High-availability Redis and PostgreSQL
- Persistent Neo4j and Qdrant clusters
- Kubernetes autoscaling and Karpenter node provisioning
- End-to-end observability with OpenTelemetry, Prometheus, and Grafana

### Use Cases

- Large-scale document ingestion
- High-concurrency agentic RAG workloads
- Distributed LLM inference
- Enterprise Kubernetes deployments
- Production-scale benchmarking and performance testing

### Branches

| Branch | Description |
|---------|-------------|
| `dev/cost-optmized` | Cost-optimized deployment |
| `dev/aws-full-scale` | High-availability, enterprise-scale deployment |

> For the recommended deployment path, refer to the `main` branch.

