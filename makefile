CC = gcc
HEADERS = linalg.h datasets.h random.h
DEPS = linalg.c datasets.c random.c nn.c 
LIBS = -lm

linear_regression: linear_regression.o
	$(CC) linear_regression.o $(DEPS) $(LIBS)

logistic_regression: logistic_regression.o
	$(CC) logistic_regression.o $(DEPS) $(LIBS)

mlp: mlp.o
	$(CC) mlp.o $(DEPS) $(LIBS)

gan: gan.o
	$(CC) gan.o -O3 -mavx -g $(DEPS) $(LIBS)
