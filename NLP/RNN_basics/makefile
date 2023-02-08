src_colab = src_rnn
image = rnn_basics_image
container = rnn_basics_container

build:
	docker build -t $(image) .

run:
	docker run -v $$(pwd):/tmp/myapp -p 8888:8888 --rm --name $(container) $(image)

bash:
	docker exec -it $(container) /bin/bash

update_container_requirements:
	docker cp $(container):/tmp/requirements.txt . 

init_collab:
	cp Project/Notebooks/$(notebook).ipynb ~/Google\ Drive/Colab\ Notebooks/.
	rm -rf ~/Google\ Drive/src/$(src_colab)
	cp -r Project/src ~/Google\ Drive/src/$(src_colab)
	cp requirements_collab.txt ~/Google\ Drive/src/$(src_colab)/.

update_drive_src:
	rm -rf ~/Google\ Drive/src/$(src_colab)
	cp -r Project/src ~/Google\ Drive/src/$(src_colab)
	cp requirements_collab.txt ~/Google\ Drive/src/$(src_colab)/.

update_notebook_from_colab:
	cp ~/Google\ Drive/Colab\ Notebooks/$(notebook).ipynb Project/Notebooks/.