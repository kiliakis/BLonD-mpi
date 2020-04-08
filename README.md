# Setup and run instructions for the HBLonD CF2020 artifact

## Paper information
Paper Title: **Scale-Out Beam Longitudinal Dynamics Simulations**
Conference: Computing Frontiers 2020, ACM International Conference on Computing Frontiers.
Acceptance Date: 29 March 2020

## Artifact setup instructions
We assume that you have downloaded and extracted the artifact, let us call it `hblond-cf2020-ae`. There are two ways to run the artifact: 
- Within a docker image, mainly to test the functionality of the artifact,  since HBLonD is meant to be executed on a multi-node cluster. 
- In an MPI configured cluster, to evaluate the functionality but also the performance and scalability of HBLonD.

## Docker image

1. These instructions target unix based systems (Linux & MacOS). However, docker is a cross-platform software that supports Windows too.  

2. Install the docker software: [https://docs.docker.com/install/](https://docs.docker.com/install/) . We recommend installing the Docker Community Edition. 

3. Optionally, you can add your user to the docker group to avoid having to run every docker command with root privileges: 
`sudo usermod -aG docker $USER` 

4. Change your working directory to the extracted artifact folder.
`cd hblond-cf2020-ae`

5. At first, build the docker image running the command:
`docker build -t hblond:1.0 -f ./Dockerfile ./`
This could take a few minutes to complete. 

6. Then to run the image, execute the command:
`docker run -it --rm --cpus=4 hblond:1.0`

6. You can test that everything is correctly set-up by running a test experiment:
	- `python __EXAMPLES/main_files/EX_01_Acceleration.py --turns 100`

6. A summary of all the command line options can be printed with the --help option:
	- `python __EXAMPLES/main_files/EX_01_Acceleration.py --help`

6. The following three real-world simulation scenarios are provided in the `__EXAMPLES/main_files` directory:
	- `LHC_main.py` (LHC testcase)
	- `SPS_main.py` (SPS testcase)
	- `PS_main.py` (PS testcase)
7. There, you can generate the figures found in the CF2020 paper from the reference data by 
running: 
`python scipts/drivers/all_cf2020_experiments.py`

8. You can run some predefined experiments with:
	```bash
	python scripts/drivers/driver.py --environment=local --action=scan --testcases lhc sps ps --output ./results
	```

9. Then extract the saved results with:
	```bash
	python scripts/drivers/driver.py --environment=local --action=extract --testcases lhc sps ps --output ./results
	```

10. Finally plot the data with:
	```bash
	python scripts/drivers/driver.py --environment=local --action=plot --testcases lhc sps ps --output ./results
	```

11. You can customize the configuration files in `scripts/scan/{lhc,sps,ps}_configs.yml` and repeat the three previous steps.

12. Or you can test interactively  the test-cases found in `__EXAMPLES/main_files` with:
`mpirun -n 4 python __EXAMPLES/main_files/LHC_main.py --turns 1000 --particles 1000000 --bunches 12`

## Cluster environment

Since every cluster has its own configuration, installed libraries, user permissions, scheduling system, etc. it is impossible to automate the setup process. Instead, we are offering detailed installation instructions.

### Setting up
1. Extract the artifact file in a directory (`hblond-cf2020-ae`).
1. Set some environment flags that will be used later:
	```bash
	export ARTIFACT_DIR = "$HOME/path/to/hblond-cf2020-ae/"
	export INSTALL_DIR = "$ARTIFACT_DIR/install"
	export BUILD_DIR = "$ARTIFACT_DIR/HBLonD"
	export PYTHONPATH = "$ARTIFACT_DIR/pymodules:$BUILD_DIR:$PYTHONPATH"
	```
1. Make sure the necessary software is installed: 
	- gcc >= 4.8
	- python >= 3.5 and python-pip
	- an MPI library (e.g. mpich, openmpi, mvapich)
	- The FFTW3 library  [http://www.fftw.org/](http://www.fftw.org/), either using the package manager (libfftw3-dev). 
	- Or you can compile FFTW3 from source by running the following commands:
	```bash
	mkdir -p $INSTALL_DIR
	cd $INSTALL_DIR
	echo "Installing FFTW3."
 	wget http://www.fftw.org/fftw-3.3.8.tar.gz
 	tar -xzvf fftw-3.3.8.tar.gz
 	cd fftw-3.3.8
 	./configure --enable-threads --with-our-malloc --disable-fortran --enable-shared
 	 make -j2 && make install
	```
1. Then install the necessary python packages:
	```bash
	cd BUILD_DIR
	pip install -r requirements.txt
	```
1. Finally compile the core library:
	```bash
	python blond/compile.py -p --with-fftw --with-fftw-threads --with-fftw-lib=$INSTALL_DIR/lib --with-fftw-header=$INSTALL_DIR/include
	```
1. If the login node is a different CPU type than the execution node, you will have to compile the library in the execution node, by modifying the script `scripts/other/batch-setup.sh` appropriately. 

1. Then you can test that everything is correctly set-up by running a test experiment:
	- `python __EXAMPLES/main_files/EX_01_Acceleration.py --turns 100`
1. A summary of all the command line options can be printed with the --help option:
	- `python __EXAMPLES/main_files/EX_01_Acceleration.py --help`
1. The following three real-world simulation scenarios are provided in the `__EXAMPLES/main_files` directory:
	- `LHC_main.py` (LHC testcase)
	- `SPS_main.py` (SPS testcase)
	- `PS_main.py` (PS testcase)
1. Our scripts assume a `slurm` scheduler. If the cluster you are using has a different scheduling system, you need to edit accordingly the file: 

1. There, you can generate the figures found in the CF2020 paper from the reference data by 
running: 
`python scipts/drivers/all_cf2020_experiments.py`

1. You can run some predefined experiments with:
	```bash
	python scripts/drivers/driver.py --environment=cluster --action=scan --testcases lhc sps ps --output ./results
	```

1. Then extract the saved results with:
	```bash
	python scripts/drivers/driver.py --environment=cluster --action=extract --testcases lhc sps ps --output ./results
	```

1. Finally plot the data with:
	```bash
	python scripts/drivers/driver.py --environment=cluster --action=plot --testcases lhc sps ps --output ./results
	```

1. You can customize the configuration files in `scripts/scan/{lhc,sps,ps}_configs.yml` and repeat the three previous steps.

## Developers and Contact info
Don't hesitate to contact us should you need further support:
- Konstantinos Iliakis (konstantinos (dot) iliakis (at) cern (dot) ch)
- Helga Timko (helga (dot) timko (at) cern (dot) ch)
- Sotirios Xydis (sxydis (at) hua (dot) gr)
- Dimitrios Soudris (dsoudris (at) microlab (dot) ntua (dot) gr)
