# Gerais
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Leitura das imagens e tratamento dos recortes
from glob import glob
import cv2
from skimage.io import imread, imread_collection
from skimage.measure import label, regionprops
from skimage.segmentation import join_segmentations

"""
Objetivo: separar áreas das leishmanias agrupadas

- Trabalha com as máscaras individuais

- Devido a sobreposição das leishmanias, algumas leishmanias tem sua área reduzida.

- Avaliei o algoritmo nas 78 imagens para remover regiões com menos de 50% da área mínima de cada leishmania. 

- O Algoritmo obteve êxito em todas as imagens

- Obteve melhores resultados em comparação com a abordagem de utilizar máscaras gerais, pois trabalha com as máscaras individuais

"""

# ################## VARIÁVEIS ################## #
dict_labels_geral = {}
area_minima = 0
qnt_labels_last_crop = 0
mask_geral, label_mask, region_area = [], [], []
PATH_IMAGENS, PATH_MASK, PATH_INDIVIDUAL = "", "", ""
IMG_ALTURA, IMG_LARGURA, IMG_CANAIS, IMG_PASSO_INICIAL, IMG_PASSO_DINAMICO, IMG_PASSO, PROPORCAO_MIN = 0, 0, 0, 0, 0, 0, 0.0
dataset = pd.DataFrame(columns=['img', 'mask', 'img_id', 'label'])
# ################## VARIÁVEIS ################## #

# Plota imagens
def exibe3imagens(img1, img2, img3):
    fig, ax = plt.subplots(1, 3, figsize=(25, 15))
    ax = ax.ravel()

    ax[0].imshow(img1, cmap="gray")
    ax[0].set_title("Recorte RGB")
    
    ax[1].imshow(img2, cmap="gray")
    ax[1].set_title("Máscara")
    
    ax[2].imshow(img3, cmap="gray")
    ax[2].set_title("Ajuste Máscara")


############################## Ler todas as máscaras da imagem correspondente #############################
def ler_mascaras(id_):
    
    # Carrega todas as máscaras da imagem
    masks = glob(PATH_INDIVIDUAL + id_[0:-4] + "-mask*.png")

    # Faz o ordenamento das máscaras
    masks = sorted(masks)

    # Ler todas as máscaras
    mascaras = imread_collection(masks)
    
    return mascaras


############################## Retorna labels de todas as mascaras #############################
def retorna_todos_labels(mascaras):
    
    # Tamanho da região para a primeira máscara da imagem
    label_mask = label(mascaras[0])
    region = regionprops(label_mask)[0] # Pega a região da mascara
    region_area.append(region.area)

    # Tamanho da região para todas as demais máscaras
    for m_ in mascaras[1:]:
        label_ = label(m_)
        region = regionprops(label_)[0]
        region_area.append(region.area)
        label_mask = join_segmentations(label_mask, label_)
    
    return label_mask

############# Seleciona os labels com área maior que a área mínima para excluir intersecções ############
def delete_intersection(label_mask):
    
    # Pega todos os labels unidos na máscara label_mask
    for region in regionprops(label_mask):

        if (region.area >= area_minima): # Remover intersecções de labels
            
            # Adiciona o label e sua área em um dicionário
            dict_labels_geral[region.label] = [region.area, region.bbox, region.image]
            
            
############################# Função que remove as leishmanias das bordas do recorte #############################
def remove_leishmania_recorte(start_h, end_h, start_w, end_w):
    
    global IMG_PASSO, qnt_labels_last_crop
    
    # Pega os labels do recorte
    # label_mask = labels de todas as máscaras individuais
    labels_recorte = regionprops(label_mask[start_h:end_h,start_w:end_w])
    qnt_labels_last_crop = len(labels_recorte)
    
    # Testa se há label no recorte
    if (qnt_labels_last_crop > 0):
        
        # Caso há leishmania, diminui o tamanho do passo para gerar mais imagens
        IMG_PASSO = IMG_LARGURA // 8
        
        # Imagem que é submetida a exclusão de leishmania
        temp_mask = mask_geral.copy()
        corte = False
        list_removed = [] # Lista de labels removidos do recorte
        
        # Pega o label, area e bbox da região de recorte e ordena por area
        dict_labels_recorte = {region.label: [region.area, region.bbox] for region in labels_recorte}

        # Itera sobre todos os labels do recorte
        for k, v in dict_labels_recorte.items():

            region_original = dict_labels_geral.get(k)
            
            # Verifica se o label não é intersecção
            if (region_original != None):

                # Pega a área e as coordenadas dos labels da região original
                area_region = region_original[0]
                minr, minc, maxr, maxc = region_original[1]

                # Verifica se a área da leishmania diminuiu no recorte
                if (v[0] < area_region):
                    proporcao = v[0] / area_region

                    # Se o recorte pegar menos que PROPORCAO_MIN da leishmania --> Ação: EXCLUI LEISHMANIA
                    if (proporcao < PROPORCAO_MIN):
                        temp_mask[minr:maxr, minc: maxc] = False # Remove leishmania
                        list_removed.append(k) # adiciona o label na lista de removidos
                        corte = True

        # Trata as demais leishmanias dentro do recorte e que NÃO FORAM CORTADAS
        if (corte):
            
            # Verifica se há mais algum label, além dos removidos no recorte
            if(len(dict_labels_recorte) > len(list_removed)):
                
                list(map(dict_labels_recorte.pop, list_removed)) # remove os labels excluídos

                for k, v in dict_labels_recorte.items():

                    region_original = dict_labels_geral.get(k)

                    # Verifica se o label não é intersecção
                    if (region_original != None):

                        # Pega as coordenadas dos labels da região original não removidos
                        minr, minc, maxr, maxc = region_original[1]
                        
                        # soma as imagens binárias (união)
                        temp_mask[minr:maxr, minc: maxc] += region_original[2] 
                        
            else:
                # Restaura o tamanho do passo caso não tenha mais leishmanias, após o corte
                IMG_PASSO = IMG_PASSO_INICIAL
                
        # Imprime a máscara original e a máscara após remoção das leishmanias
        # exibe3imagens(label_mask[start_h:end_h,start_w:end_w], mask_geral[start_h:end_h,start_w:end_w], temp_mask[start_h:end_h,start_w:end_w])
        
        # retorna retorna a máscara em float e a classe dela
        if (temp_mask[start_h:end_h,start_w:end_w].max() == True):
            return np.expand_dims(temp_mask[start_h:end_h,start_w:end_w].astype(float), axis=-1), 1 # converte para float e expande a dimensão
        else:
            return np.zeros((IMG_ALTURA, IMG_LARGURA, 1), dtype=np.float32), 0
        
    else:
        # Ajusta o passo para pegar menos recortes negativos
        IMG_PASSO = IMG_LARGURA
        
        return np.zeros((IMG_ALTURA, IMG_LARGURA, 1), dtype=np.float32), 0
    
    
############################# Recorta as imagens #############################
def recorta_imagens(img, mask_geral, img_cont):
    
    global dataset, IMG_PASSO, IMG_PASSO_INICIAL, qnt_labels_last_crop
    
    # Variáveis recorte
    start_h = 0
    start_w = 0
    end_h = IMG_ALTURA
    end_w = IMG_LARGURA
    qnt_labels_last_line, qnt_labels_last_crop = 0, 0
    IMG_PASSO = IMG_PASSO_INICIAL

    # Enquanto for possível fazer recorte na ALTURA
    while((start_h + IMG_ALTURA) <= img.shape[0]):
            
        # Enquanto for possível fazer recorte na LARGURA
        while((start_w + IMG_LARGURA) <= img.shape[1]): 

            # Recorte imagens e máscaras
            img_recorte = img[start_h:end_h,start_w:end_w,:IMG_CANAIS]
            mask_recorte, label = remove_leishmania_recorte(start_h, end_h, start_w, end_w)
            dataset = dataset.append({'img':img_recorte, 'mask': mask_recorte, 'img_id': img_cont, 'label': label}, ignore_index=True)

            # Atualiza variáveis LARGURA
            start_w = start_w + IMG_PASSO
            end_w = end_w + IMG_PASSO
            
            # Atualiza a qnt de recortes na linha atual
            qnt_labels_last_line += qnt_labels_last_crop

            # Sai da LARGURA quando não há espaço para mais um recorte completo
            if ((start_w + IMG_LARGURA) > img.shape[1]):
                
                # Só pega os pixels da margem direita se tiver leishmania identificada no ultimo recorte
                if (qnt_labels_last_crop) > 0:
                    
                    # Pega os últimos pixels da margem direita da imagem
                    img_recorte = img[start_h:end_h,-IMG_LARGURA:,:IMG_CANAIS]
                    mask_recorte, label = remove_leishmania_recorte(start_h, end_h, img.shape[1]-IMG_LARGURA, img.shape[1])
                    dataset = dataset.append({'img':img_recorte, 'mask': mask_recorte, 'img_id': img_cont, 'label': label}, ignore_index=True)
                    
                    # Atualiza a qnt de recortes na linha atual
                    qnt_labels_last_line += qnt_labels_last_crop

                # Sai da LARGURA
                break

        # Reinicia o passo caso tenha leishmania na ultima linha processada
        if (qnt_labels_last_line > 0):
            IMG_PASSO = IMG_PASSO_INICIAL
        
        # Reinicia as variáveis LARGURA
        start_w = 0
        end_w = IMG_LARGURA

        # Atualiza variáveis ALTURA
        start_h = start_h + IMG_PASSO
        end_h = end_h + IMG_PASSO

        # Sai da ALTURA quando não há espaço para mais um recorte completo
        if ((start_h + IMG_ALTURA) > img.shape[0]):

            # Só pega os últimos pixels do rodapé quando alguma leishmania é identificada na ultima linha
            if (qnt_labels_last_line) > 0:

                # Pega os últimos pixels do rodapé da imagem
                # Enquanto for possível fazer recorte na LARGURA
                while((start_w + IMG_LARGURA) <= img.shape[1]): 
                    img_recorte = img[-IMG_ALTURA:,start_w:end_w,:IMG_CANAIS]
                    mask_recorte, label = remove_leishmania_recorte(img.shape[0]-IMG_ALTURA, img.shape[0], start_w, end_w)
                    dataset = dataset.append({'img':img_recorte, 'mask': mask_recorte, 'img_id': img_cont, 'label': label}, ignore_index=True)

                    # Atualiza variáveis LARGURA
                    start_w = start_w + IMG_PASSO
                    end_w = end_w + IMG_PASSO

                    # Sai da LARGURA quando não há espaço para mais um recorte completo
                    if ((start_w + IMG_LARGURA) > img.shape[1]):
                        
                        # Só pega os últimos pixels do canto inferior direito se tiver leishmania identificada no ultimo recorte
                        if (qnt_labels_last_crop) > 0:

                            # Pega os últimos pixels do canto inferior direito da imagem
                            img_recorte = img[-IMG_ALTURA:,-IMG_LARGURA:,:IMG_CANAIS]
                            mask_recorte, label = remove_leishmania_recorte(img.shape[0]-IMG_ALTURA, img.shape[0], img.shape[1]-IMG_LARGURA, img.shape[1])
                            dataset = dataset.append({'img':img_recorte, 'mask': mask_recorte, 'img_id': img_cont, 'label': label}, ignore_index=True)

                        # Sai da LARGURA
                        break

            # Sai da ALTURA
            break
        
        # Reinicia o contador de labels na ultima linha processada
        qnt_labels_last_line = 0
            
# ############################# Carrega as imagens e realiza os recortes ############################# #
def load_crop_images(PATH_IMAGENS_, 
                           PATH_MASK_, 
                           PATH_INDIVIDUAL_, 
                           IMG_ALTURA_, 
                           IMG_LARGURA_, 
                           IMG_CANAIS_, 
                           IMG_PASSO_INICIAL_, 
                           IMG_PASSO_DINAMICO_, 
                           PROPORCAO_MIN_):
    
    global PATH_IMAGENS, PATH_MASK, PATH_INDIVIDUAL, IMG_ALTURA, IMG_LARGURA, IMG_CANAIS, IMG_PASSO_INICIAL, IMG_PASSO_DINAMICO, PROPORCAO_MIN
    global region_area, dict_labels_geral, area_minima, mask_geral, label_mask, dataset
    
    
    # Variáveis globais recebem os parâmetros da função
    PATH_IMAGENS = PATH_IMAGENS_
    PATH_MASK = PATH_MASK_
    PATH_INDIVIDUAL = PATH_INDIVIDUAL_
    IMG_ALTURA = IMG_ALTURA_
    IMG_LARGURA = IMG_LARGURA_
    IMG_CANAIS = IMG_CANAIS_
    IMG_PASSO_INICIAL = IMG_PASSO_INICIAL_
    IMG_PASSO_DINAMICO = IMG_PASSO_DINAMICO_
    PROPORCAO_MIN = PROPORCAO_MIN_
    
    # Pega o nome de todas as imagens em ordem alfabética
    imagens_ids = sorted(next(os.walk(PATH_IMAGENS))[2])
    print("Total de Imagens: ", len(imagens_ids))
    
    for img_cont, id_ in tqdm(enumerate(imagens_ids), total = len(imagens_ids)):

        # Carrega as imagens em LUV
        img = cv2.imread(PATH_IMAGENS + id_)[:,:,:IMG_CANAIS]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV) # Converte para LUV
        #img = img[:,:,2] # Canal V do LUV
        img = img / 255.0
        
        # Carrega a máscara geral
        id_ = id_.lower().replace(" ", "") # Pasta com 78 imagens
        mask_geral = cv2.imread(PATH_MASK + id_[0:-4] + "-mask.png", 0) # ler em escala de cinza
        mask_geral = mask_geral > 0 # binário

        # Armazena a área das regiões dos labels para pegar a área mínima de cada imagem
        region_area = []

        # Armazena o label, sua área e seu bbox, após remoção das intersecções de labels baseado na área mínima
        dict_labels_geral = {}

        # Ler todas as máscaras da imagem correspondente
        mascaras = ler_mascaras(id_)

        # Retorna labels de todas as mascaras
        label_mask = retorna_todos_labels(mascaras)

        # Define área mínima para excluir intersecções entre os labels
        # Devido a sobreposição de labels, algumas áreas tem o tamanho reduzido
        # Só considera amastigotas reduzidas até 50% da menor amastigota da máscara inteira
        # Cheguei a 0.5 após testes em todas as imagens
        area_minima = int(min(region_area) * 0.50) 

        # Seleciona os labels que possuem área maior que a área mínima
        # Objetivo: excluir intersecções entre os labels
        delete_intersection(label_mask)

        # Recorta as imagens e máscaras
        recorta_imagens(img, mask_geral, img_cont)
    
    # Convertendo tipos de dados para melhorar uso de memória
    dataset.img_id = dataset.img_id.astype('category')
    dataset.label = dataset.label.astype('category')
    
    return dataset