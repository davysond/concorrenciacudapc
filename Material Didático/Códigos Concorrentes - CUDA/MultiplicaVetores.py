import numpy as np
from timeit import default_timer as timer 
from numba import vectorize

@vectorize(["float32(float32,float32)"], target='cuda')
def multiplicaVetoresCUDA(a, b):
	return a * b

def multiplicaVetoresCPU(a, b):
	return a * b

def main():
	N = 10000000 #tamanho por array inserido
	A = np.ones(N, dtype = np.float32)
	B = np.ones(N, dtype = np.float32)
	C = np.ones(N, dtype = np.float32)
	D = np.ones(N, dtype = np.float32)

	### CPU ###

	tempoInicialCPU = timer()
	C = multiplicaVetoresCPU(A, B)
	tempoVetorizacaoCPU = timer() - tempoInicialCPU

	print("C[:6] = " + str(C[:6]))
	print("C[-6] = " + str(C[-6:]))

	print(f'A multiplicação, utilizando a CPU, {tempoVetorizacaoCPU} segundos')

	print('\n')

	### CUDA ### 

	tempoInicialCuda = timer()
	C = multiplicaVetoresCUDA(A, B)
	tempoVetorizacaoCuda = timer() - tempoInicialCuda

	print("D[:6] = " + str(D[:6]))
	print("D[-6] = " + str(D[-6:]))

	print(f'A multiplicação, utilizando virtualização CUDA, levou {tempoVetorizacaoCuda} segundos')
	
main()