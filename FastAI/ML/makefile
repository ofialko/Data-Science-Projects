build:
	docker build -t ml_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name ml_container ml_image

bash:
	docker exec -it ml_container /bin/bash

update_toml:
	docker cp ml_container:/tmp/pyproject.toml . 
	docker cp ml_container:/tmp/poetry.lock .

upload_to_collab:
	cp Project/Notebooks/$(notebook).ipynb ~/Google\ Drive/Colab\ Notebooks/.
	rm -rf ~/Google\ Drive/data/data_$(notebook)
	cp -r Project/data ~/Google\ Drive/data/data_$(notebook)
	rm -rf ~/Google\ Drive/src/src_$(notebook)
	cp -r Project/src ~/Google\ Drive/src/src_$(notebook)
	cp requirements.txt ~/Google\ Drive/src/src_$(notebook)/.

update_src_drive:
	rm -rf ~/Google\ Drive/src/src_$(notebook)
	cp -r Project/src ~/Google\ Drive/src/src_$(notebook)

update_from_colab:
	cp ~/Google\ Drive/Colab\ Notebooks/$(notebook).ipynb Project/Notebooks/.