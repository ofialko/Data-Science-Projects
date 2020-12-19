build:
	docker build -t species_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name species_container species_image

bash:
	docker exec -it species_container /bin/bash

cp_poetry:
	docker cp species_container:/tmp/pyproject.toml .