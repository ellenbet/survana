### Biomarker and survival analysis software for thesis.

<img width="562" height="768" alt="image" src="https://github.com/user-attachments/assets/6bebe705-bbea-4dfd-a193-0d1f2fe63c58" />

How to use and build:
1. Clone repo
2. Run ``uv sync`` in terminal
3. Run ``pre-commit install`` in terminal
4. Make sure to have some helpful extensions installed: mypy, isort, black, flake8
5. Play around (test2)


### Docker
What and why?

Docker solves the problems that occur from running a project on two different machines. Docker containers are self-contained to run as they are, and isolated with kernel isolation features. Using a Docker container is comparable to VM's, but smaller with better performance.

Using Docker:
- Docker Desktop
    - Docker deamon
    - CLI
    - GUI

Container vs. images:
One image: template for container; filesystem, user, commands etc. Recipie for running.
Two containers: the actual groups that follow image-specified instructions. Many containers from one image.

When running a container, you can use a tag to specify which image it is based on.
