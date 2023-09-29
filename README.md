# Instruções

**Passo 1** - gerar arquivos de dados executando `python gen_data.py`. Os arquivos gerados na pasta data/ são:

* visits.csv - arquivo base de visitas
* visits_normalized_masked-[% de dados ausentes por atributo.csv] - arquivo transformado de visitas, com conversão de variáveis categóricas em variáveis dummy e normalização das propensões de compra
* products.json - arquivo com os dados dos produtos gerados aleatoriamente (para criaçãodas visitas)
* products_stats.csv - arquivo com as propensões de compra originais dos produtos (antes da normalização)

**Passo 2** - executar a simulação do sistema de recomendação executando `python recommenders_ML.py`.

O arquivo gerado é o `results.csv`, contendo em um único arquivo todas as simulações para todos os cenários.

O layout do arquivo results.csv é:

* visit_id - identificador da visita
* period - identificador do período
* selected_product - produto convertido originalmente
* predicted - produto recomendado (pelo algoritmo)
* algorithm - algoritmo de classificação utilizado na recomendação (naive bayes, rede neural, etc)
* model_type - tipo do modelo utilizado (regular ou alternate)
* model_version - indicador de versão do modelo; sempre que um novo modelo é criado, um número distinto é associado a esse modelo
* cycle_duration - duração do ciclo do modelo (número de visitas entre o início do processo regular de recomendação e o fim do período de teste A/B)
* ab_duration_pct - duração do teste A/B em relação à duração do ciclo (ex. 0.3 indica que o teste A/B tem 0.3 x cycle_duration visitas)
* alternate_model_pct - porcentagem do número de visitas (com recomendações) utilizadas para validação do modelo
* masking - porcentagem de dados ausentes por atributo no arquivo de visita

Nota: no momento em que os scripts foram desenvolvidos, o processo de introdução de dados ausentes era chamado "masking"; o nome original foi mantido no código, pelo menos até esta versão.
