ARG BASE_IMAGE_TAG
FROM --platform=$BUILDPLATFORM eclipse-temurin:11 AS builder

RUN curl https://raw.githubusercontent.com/sbt/sbt/develop/sbt > /sbt && chmod +x /sbt && export PATH=/:$PATH

RUN mkdir -p /tmp/project/project
WORKDIR /tmp/project

# prime sbt for cache
COPY project/build.properties /tmp/project/project/
RUN /sbt exit

# warmup caches and create dependency jar, if build doesn't change, we will use caches in the next run
COPY build.sbt /tmp/project/
COPY project/* /tmp/project/project/

RUN /sbt update "show web/assemblyPackageDependency"

COPY src /tmp/project/src

RUN /sbt "show web/assembly"

FROM eclipse-temurin:17-jre

COPY --from=builder /tmp/project/web/target/scala-3.3.0/deps.jar /deps.jar
COPY --from=builder /tmp/project/web/target/scala-3.3.0/app.jar /app.jar

EXPOSE 8087/tcp

WORKDIR /

CMD ["java", "-cp", "/app.jar:/deps.jar", "net.virtualvoid.llama2.web.Llama2WebMain"]
