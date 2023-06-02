# list all available commands
default:
  just --list

# clean all build, python, and lint files
clean:
	rm -fr build
	rm -fr docs/_build
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov
	rm -fr .pytest_cache
	rm -fr .mypy_cache

# install with all deps
install:
	pip install -e '.[lint,test,docs,dev]'

# lint, format, and check all files
lint:
	pre-commit run --all-files

# run tests
test:
	pytest semantic_navigator/tests

# run lint and then run tests
build:
	just lint
	just test

# generate Sphinx HTML documentation
generate-docs:
	rm -f docs/semantic_navigator*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs semantic_navigator **/tests
	python -msphinx "docs" "docs/_build"


# Generate project URI for browser opening
# We replace here to handle windows paths
# Windows paths are normally `\` separated but even in the browser they use `/`
# https://stackoverflow.com/a/61991869
project_uri := if "os_family()" == "unix" {
	justfile_directory()
} else {
	replace(justfile_directory(), "\\", "/")
}

# generate Sphinx HTML documentation and serve to browser
serve-docs:
	just generate-docs
	python -mwebbrowser -t "file://{{project_uri}}/docs/_build/index.html"

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname| head -n 1)"

# release a new version
release:
	git push --follow-tags

# update this repo using latest cookiecutter-py-package
update-from-cookiecutter:
	pip install cookiecutter
	cookiecutter gh:evamaxfield/cookiecutter-py-package --config-file .cookiecutter.yaml --no-input --overwrite-if-exists --output-dir ..

###############################################################################
# infra

# get and store user
USER := env_var("USER")

# default params
default_region := "us-central1"
default_key_dir := ".keys/"
default_key := ".keys/sem-nav-dev.json"
default_project := "sem-nav-eva-005"

# run gcloud login
login:
    gcloud auth login
    gcloud auth application-default login

# generate a service account JSON
gen-key project=default_project:
    mkdir -p {{default_key_dir}}
    rm -rf {{default_key_dir}}{{project}}.json
    gcloud iam service-accounts create {{project}} \
        --description="Dev Service Account for {{USER}}" \
        --display-name="{{project}}"
    gcloud projects add-iam-policy-binding {{project}} \
        --member="serviceAccount:{{project}}@{{project}}.iam.gserviceaccount.com" \
        --role="roles/owner"
    gcloud iam service-accounts keys create {{default_key_dir}}{{project}}.json \
        --iam-account "{{project}}@{{project}}.iam.gserviceaccount.com"
    @ echo "----------------------------------------------------------------------------"
    @ echo "Sleeping for fifteen seconds while resources set up"
    @ echo "----------------------------------------------------------------------------"
    sleep 15
    cp -rf {{default_key_dir}}{{project}}.json {{default_key}}
    @ echo "----------------------------------------------------------------------------"
    @ echo "Be sure to update the GOOGLE_APPLICATION_CREDENTIALS environment variable."
    @ echo "----------------------------------------------------------------------------"

# create a new gcloud project and generate a key
init project=default_project:
    gcloud projects create {{project}} --set-as-default
    @ echo "----------------------------------------------------------------------------"
    @ echo "Follow the link to setup billing for the created GCloud account."
    @ echo "https://console.cloud.google.com/billing/linkedaccount?project={{project}}"
    @ echo "----------------------------------------------------------------------------"
    just gen-key {{project}}

# enable gcloud services
enable-services:
    gcloud services enable cloudresourcemanager.googleapis.com
    gcloud services enable cloudfunctions.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com \
        run.googleapis.com

# deploy the web app
deploy project=default_project region=default_region:
    just enable-services
    -gsutil mb gs://{{project}}
    gsutil cors set cors.json gs://{{project}}
    gsutil defacl ch -u AllUsers:R gs://{{project}}
    gcloud builds submit --tag gcr.io/{{project}}/semanticnavigator
    gcloud run deploy semanticnavigator \
        --image gcr.io/{{project}}/semanticnavigator \
        --region {{region}} \
        --allow-unauthenticated \
        --memory 4Gi

# fully teardown project
destroy project:
    gcloud projects delete {{project}}
    rm -f {{default_key_dir}}{{project}}.json

# build docker image locally
build-docker:
	docker build --tag semantic-navigator {{justfile_directory()}}

# run docker image locally
run-docker:
	docker run --rm -p 8080:8080 -e PORT=8080 semantic-navigator