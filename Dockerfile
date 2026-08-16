ARG go_build_image=golang
ARG go_build_tag=1.24-bookworm

ARG app_image=debian
ARG app_tag=bookworm-slim

FROM ${go_build_image}:${go_build_tag} AS go_build
RUN mkdir -p /build
WORKDIR /build
ADD . /build
RUN go build -v -trimpath -ldflags "-s -w" -o qwen38-rp .

FROM ${app_image}:${app_tag}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/* \
    && groupadd -r qwen38rp && useradd -r -g qwen38rp qwen38rp
COPY --from=go_build /build/qwen38-rp /usr/bin/qwen38-rp
RUN chown qwen38rp:qwen38rp /usr/bin/qwen38-rp
USER qwen38rp

EXPOSE 9000

ENTRYPOINT ["/usr/bin/qwen38-rp"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1
