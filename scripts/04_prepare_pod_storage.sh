#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="$ROOT_DIR/k8s"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is not installed or not on PATH" >&2
  exit 1
fi

if ! kubectl cluster-info >/dev/null 2>&1; then
  echo "No Kubernetes cluster is currently reachable from this environment" >&2
  exit 1
fi

kubectl apply -f "$K8S_DIR/chai-downloads-pvc.yaml"
kubectl apply -f "$K8S_DIR/pod-template.yaml"

echo "Persistent volume claim created and pod template updated with CHAI_DOWNLOADS_DIR=/workspace/chai_downloads"
