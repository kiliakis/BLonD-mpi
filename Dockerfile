#FROM vauxoo/odoo-80-image-shippable-auto

FROM ubuntu:18.04

ENV HOME=/root
ENV MPI_ENV_DIR=$HOME/mpi-env
ENV INSTALL_DIR=${MPI_ENV_DIR}/install
ENV REPO_SLUG=HBLonD
ENV BUILD_DIR=${MPI_ENV_DIR}/${REPO_SLUG}
ENV VIRTUAL_ENV=/root/venv

COPY ./files/.bashrc $HOME/

RUN	echo "export PYTHONPATH=$BUILD_DIR:$MPI_ENV_DIR/pymodules:$PYTHONPATH" >> $HOME/.bashrc && \
	echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH" >> $HOME/.bashrc && \
	echo "export INSTALL_DIR=$INSTALL_DIR" >> $HOME/.bashrc && \
	echo "export BUILD_DIR=$BUILD_DIR" >> $HOME/.bashrc && \
	echo "export MPI_ENV_DIR=$MPI_ENV_DIR" >> $HOME/.bashrc && \
	echo "source $VIRTUAL_ENV/bin/activate" >> $HOME/.bashrc


RUN apt-get update && \
        apt-get install -y software-properties-common && \
        apt-add-repository -y "ppa:ubuntu-toolchain-r/test" && \
        apt-get update -y  && \
        apt-get -yq --no-install-suggests --no-install-recommends install apt-utils build-essential mpich libmpich-dev wget git python3 python3-dev python3-virtualenv python3-pip libfftw3-dev

#RUN	mkdir -p $INSTALL_DIR && \
#	cd $INSTALL_DIR && \
#	echo "Installing FFTW3." && \
# 	wget http://www.fftw.org/fftw-3.3.8.tar.gz && \
# 	tar -xzvf fftw-3.3.8.tar.gz > /dev/null && \
# 	cd fftw-3.3.8 && \
# 	./configure --enable-threads --with-our-malloc --disable-fortran --enable-shared > /dev/null && \
# 	 make -j2 > /dev/null && make install > /dev/null

RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN git clone --branch=artifact https://github.com/kiliakis/BLonD-mpi.git ${BUILD_DIR} \
    && cd ${BUILD_DIR} || true

WORKDIR ${BUILD_DIR}
COPY ./files/pymodules $MPI_ENV_DIR/pymodules
COPY ./files/input_files $BUILD_DIR/__EXAMPLES/input_files

    
RUN cd $BUILD_DIR && \
	python3 -m pip install --upgrade pip setuptools wheel pytest && \
	python3 -m pip install numpy scipy matplotlib future h5py mpi4py

RUN python3 blond/compile.py -p --with-fftw --with-fftw-threads


#RUN /bin/bash -c "source ${REPO_REQUIREMENTS}/virtualenv/python3.6/bin/activate && source /rvm_env.sh && /before_install && /install"

#ENTRYPOINT /entrypoint.sh