CC = clang
CFLAGS = -Wall -Wextra -O3 -std=c23 -ffast-math -fno-finite-math-only -march=native -DNDEBUG
#CFLAGS = -Wall -O0 -std=c23 -g -fsanitize=address -fsanitize=thread
LDFLAGS = -lm

DEMO_SRC := demo.c mnist.c
SRC := demo.c mnist.c
HEADERS := mnist.h 

ASTYLE_OPTS := --style=kr \
               --indent=spaces=8 \
               --pad-oper \
               --pad-comma \
               --unpad-brackets \
               --squeeze-ws

.PHONY: all clean fmt

all: demo 

demo: $(DEMO_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(DEMO_SRC) $(LDFLAGS)

clean:
	rm -f demo 

fmt:
	astyle $(ASTYLE_OPTS) $(SRC) $(HEADERS)

