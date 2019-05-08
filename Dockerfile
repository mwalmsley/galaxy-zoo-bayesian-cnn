FROM python:3.6

WORKDIR your/path/to/this/repo

# add requirements.txt separately to allow the build to be cached and re-used
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . .

# Remove any trailing cache files to avoid confusing pytest with file locations
RUN find . -name '*.pyc' -delete