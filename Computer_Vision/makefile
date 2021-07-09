project := $(shell basename $(abspath $(dir $$PWD)))

gdrive_path = Google\ Drive

src_colab   = src_$(project)
data_colab  = data_$(project)
models_colab = models_$(project)
notebook_colab = notebook_$(project)

image = image_$(project)
container = container_$(project)

data_colab_path  = ~/$(gdrive_path)/data/$(data_colab)
src_colab_path   = ~/$(gdrive_path)/src/$(src_colab)
models_colab_path = ~/$(gdrive_path)/models/$(models_colab)
notebook_colab_path = ~/$(gdrive_path)/Colab\ Notebooks

build:
	docker build -t $(image) .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name $(container) $(image)

bash:
	docker exec -it $(container) /bin/bash

update_container_requirements:
	docker cp $(container):/tmp/requirements.txt . 

init_collab:
	mkdir -p $(data_colab_path)
	mkdir -p $(src_colab_path)
	mkdir -p $(models_colab_path)
	cp Project/Notebooks/$(notebook_colab).ipynb $(notebook_colab_path)/.
	rm -rf $(data_colab_path)
	cp -r Project/data $(data_colab_path)
	rm -rf $(src_colab_path)
	cp -r Project/src $(src_colab_path)
	rm -rf $(models_colab_path)
	cp -r Project/models $(models_colab_path)
	cp requirements_collab.txt $(src_colab_path)/.

update_drive_src:
	rm -rf $(src_colab_path)
	cp -r Project/src $(src_colab_path)
	cp requirements_collab.txt $(src_colab_path)/.

update_from_colab:
	cp $(notebook_colab_path)/$(notebook_colab).ipynb Project/Notebooks/.
	cp -r $(models_colab_path) Project/models/.


clear_colab:
	rm -rf $(data_colab_path)
	rm -rf $(src_colab_path)
	rm -rf $(models_colab_path)
	rm $(notebook_colab_path)/$(notebook_colab).ipynb
