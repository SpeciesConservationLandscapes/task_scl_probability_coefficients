FROM scl3/task_base:latest

RUN echo "http://dl-cdn.alpinelinux.org/alpine/v3.5/community" >> /etc/apk/repositories \
    && apk update \
    && apk add --upgrade --no-cache \
        # scipy build dependencies
        openssh ca-certificates openssl htop g++ make \
        libpng-dev freetype-dev libexecinfo-dev openblas-dev libgomp lapack-dev \
		libgcc libquadmath musl libgfortran lapack-dev \
        # odbc drivers
		unixodbc-dev freetds-dev
		# TODO: cleanup and remove caches etc.

RUN pip install git+https://github.com/SpeciesConservationLandscapes/task_base.git \
    && pip install numpy==1.17.3 \
    && pip install pandas==1.0.2 \
    && pip install scipy==1.3.1 \
    && pip install pyodbc==4.0.30 \
    && pip install geomet==0.2.1.post1

WORKDIR /app
COPY $PWD/src .
COPY $PWD/odbcinst.ini /etc/odbcinst.ini
