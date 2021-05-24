src_colab = src_rnn

build:
	docker build -t rnn_basics_image .

run:
	docker run -v $$(pwd):/tmp/myapp -p 8888:8888 --rm --name rnn_basics_container rnn_basics_image

bash:
	docker exec -it rnn_basics_container /bin/bash

update_toml:
	docker cp rnn_basics_container:/tmp/requirements.txt . 

upload_to_collab:
	cp Project/Notebooks/$(notebook).ipynb ~/Google\ Drive/Colab\ Notebooks/.
	rm -rf ~/Google\ Drive/src/$(src_colab)
	cp -r Project/src ~/Google\ Drive/src/$(src_colab)
	cp requirements_collab.txt ~/Google\ Drive/src/$(src_colab)/.

update_src_drive:
	rm -rf ~/Google\ Drive/src/$(src_colab)
	cp -r Project/src ~/Google\ Drive/src/$(src_colab)
	cp requirements_collab.txt ~/Google\ Drive/src/$(src_colab)/.

update_from_colab:
	cp ~/Google\ Drive/Colab\ Notebooks/$(notebook).ipynb Project/Notebooks/.