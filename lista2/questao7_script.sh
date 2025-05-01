#!/bin/bash

# Script para compilar e executar os experimentos CUDA com nvprof

# Compilar todos os programas
echo "Compilando todos os programas..."
nvcc -O3 questao7_base.cu -o questao7_base
nvcc -O3 questao7_a.cu -o questao7_a
nvcc -O3 questao7_b.cu -o questao7_b
nvcc -O3 questao7_c.cu -o questao7_c
nvcc -O3 questao7_e.cu -o questao7_e

# Função para executar um experimento com nvprof
run_experiment() {
    echo "================================================================="
    echo "Executando experimento $1:"
    echo "-----------------------------------------------------------------"
    nvprof --unified-memory-profiling per-process-device \
           --print-gpu-trace \
           --csv ./$1
    echo ""
}

# Função para executar um experimento com nvprof focado em falhas de página
run_page_fault_metrics() {
    echo "================================================================="
    echo "Métricas de falha de página para $1:"
    echo "-----------------------------------------------------------------"
    nvprof --unified-memory-profiling per-process-device \
           --metrics unified_memory_gpu_page_fault_count,unified_memory_cpu_page_fault_count \
           --csv ./$1
    echo ""
}

# Experimento Base (sem prefetch)
echo ""
echo "************************ EXPERIMENTO BASE ************************"
run_experiment questao7_base
run_page_fault_metrics questao7_base

# Experimento A (prefetch apenas do vetor a)
echo ""
echo "************************ EXPERIMENTO A ************************"
run_experiment questao7_a
run_page_fault_metrics questao7_a

# Experimento B (prefetch dos vetores a e b)
echo ""
echo "************************ EXPERIMENTO B ************************"
run_experiment questao7_b
run_page_fault_metrics questao7_b

# Experimento C (prefetch de todos os vetores)
echo ""
echo "************************ EXPERIMENTO C ************************"
run_experiment questao7_c
run_page_fault_metrics questao7_c

# Experimento E (prefetch de volta para CPU)
echo ""
echo "************************ EXPERIMENTO E ************************"
run_experiment questao7_e
run_page_fault_metrics questao7_e

echo ""
echo "======================== RESUMO DOS RESULTADOS ========================"
echo "Execute os comandos abaixo para ver apenas os tempos de cada versão:"
echo "  ./questao7_base"
echo "  ./questao7_a"
echo "  ./questao7_b"
echo "  ./questao7_c"
echo "  ./questao7_e"
echo ""
echo "Os resultados completos do nvprof estão acima." 