
########################################
# Compiler
########################################

GCC = g++

RM = rm

########################################
# FLAGS
########################################

CFLAGS = -std=c++11 -lstdc++

INCFLAG = -I"." -I"./common/includes/"

LIBFLAG = -L"./common/lib/"

FOPENCL = -framework OpenCL

FOPENGL = -lglad -lglfw -lassimp -framework opengl

CC = $(GCC) $(CFLAGS) $(FOPENCL) $(FOPENGL) $(INCFLAG) $(LIBFLAG)

########################################
# PROGRAM SPEC
########################################

program = fluid.exe

source = \
main.cpp \
Shader.cpp \
EularCamera.cpp \
Texture.cpp \
Mesh.cpp \
Model.cpp \
Primitives.cpp \
cl.cpp

object = $(source:.cpp=.o)

########################################
# BUILDING
########################################

all: $(program) $(object)

%.exe: $(object)
	$(CC) $(object) -o $@ -lm

%.o: %.cpp %.h
	$(CC) -c $< -o $@ -lm

clean: 
	$(RM) -f $(program) $(object)
