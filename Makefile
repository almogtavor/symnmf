CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

all: symnmf

symnmf: symnmf.o utils.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o utils.o -lm

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf
