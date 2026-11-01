# Build the itsfraud service. Model files are mounted at runtime (see MODEL_CARD.md), not baked in:
#   docker build -t itsfraud .
#   docker run --rm -p 8080:8080 -v "$PWD/models/v0.1.0:/models:ro" itsfraud
FROM rust:1-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release -p server

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/itsfraud /usr/local/bin/itsfraud
ENV ITSFRAUD_ARTIFACTS=/models
EXPOSE 8080
USER nobody
CMD ["itsfraud", "serve", "0.0.0.0:8080"]
