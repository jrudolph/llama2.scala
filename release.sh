#!/bin/sh

set -e
set -x

VERSION=$1

git tag -a $VERSION -m "Released $VERSION"
docker buildx build --builder kube --platform linux/amd64,linux/arm64 --tag registry.virtual-void.net/jrudolph/llama2explorer:$VERSION --push .
