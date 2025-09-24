## Docker
What and why?

Docker solves the problems that occur from running a project on two different machines. Docker containers are self-contained to run as they are, and isolated with kernel isolation features. Using a Docker container is comparable to VM's, but smaller with better performance.

Using Docker:
- Docker Desktop
    - Docker deamon
    - CLI
    - GUI

### Container vs. images:
One image: template for container; filesystem, user, commands etc. Recipie for running.
Two containers: the actual groups that follow image-specified instructions. Many containers from one image.

When running a container, you can use a tag to specify which image it is based on. Pinning by digest: better than pinning by version, digest is directly connected to image.

### Slim and Alpine images:
Slim versions are 10x smaller than default Python images. Alpine images are minimalist versions, 20x smaller than main version. Generally recommended to use either slim or alpine. Slim is debian based Linux.

### Debugging containers:
You can use GUI Exec root terminal to debug, or docker exec to execute processes on already-running containers.

### Mount:
To keep changes across containers and images, we can mount a database across different containers.

Three kinds:
- Volumes
    - Persistent
    - Newer, more features
    - Managed by docker daemon
    - -v mydata:/path/in/container
- Bind-mounts
    - Persistent
    - Older, less features
    - Mounts host file/dir into container
    - -v ./mydata:/path/in/container
- Tempfs mounts

Example:
``docker run -v ./mydata:/data python:3.12 python -c "f="data/data.txt";open(f, "a").write("Ran!\n");print(open(f).read())``

Mounting the mydata directory to the container keeps adding new data to existing data directory rather than starting everything anew. Volumes or bind-mounts? In dev, bind-mounts can be more convenient. Volumnes are often better in production settings, easily shared across containers and we can use cloud storage. Containers do not have access to host, unlike with bind-mounts.

### Custom images
Building your own images: create a project folder.

### Layers
Each command in a Dockerile creates a layer, works like an image diff. Layers are immutable, all commands create new layers. All layers are cached and tiny diffs are added as we go. Different commands have different caching rules: changing commands high up will cause a chain-effect on all commands lower down. This helps speed up the build process significantly!

Security warning: a run command that deletes something creates a new layer on top of it. Sensitive info in a image will never actually be deleted (like Git history).
