import numpy as np
from math import sqrt, pi
from utils.tables import Tables


class Transformer:
    def __init__(self, constraints: list, tables: dict):
        self.constantes = constraints
        self.tables = Tables(tables)

    def update_tables(self, tables: str):
        self.tables.tables.set_many(tables)

    def run(self, variaveis: np.ndarray):
        # print(variaveis)
        [
            conexao,
            Ke,
            S,
            Nfases,
            f,
            V1,
            V2,
            tipo,
            Dfe,
            Dal,
        ] = self.constantes
        try:
            [
                Jbt,
                Jat,
                Bm,
                Ksw,
                kt,
                Rjan,
                rel,
            ] = variaveis
        except:
            [
                _,
                Jbt,
                Jat,
                Bm,
                Ksw,
                kt,
                Rjan,
                rel,
            ] = variaveis

        # from ipdb import set_trace; set_trace()
        secundario, primario = conexao.split("-")

        Vf1, Vf2 = V1, V2

        if Nfases == 3:
            if primario.lower() == "estrela":
                Vf1 = V1 / sqrt(3)

            if secundario.lower() == "estrela":
                Vf2 = V2 / sqrt(3)

        # tabelas = Tables(self.tables)

        Et = kt * sqrt(S)       # é a tensão eficaz por espiras [V/e]
        N1 = (Vf1 * 1000) / Et
        N2 = (Vf2 * 1000) / Et

        Ac = Et / (4.44 * f * Bm) * 1e6  # é a área efetiva da coluna  [mm2]
        Abc = Ac / Ke
        numero_degraus = self.tables.numero_degraus(Abc / 1000)

        Ku, LD = self.tables.dimensoes_nucleo(numero_degraus)
        # import ipdb; ipdb.set_trace()

        # LD é um vetor que contem todos os valores existentes na tabela 2.4
        LD = np.asarray(LD, np.float64)

        So = Abc / Ku           # Seção circular circunscrita
        dc = 2 * sqrt(So / pi)  # é o diâmetro da coluna do núcleo

        L = LD * dc
        teta = np.arccos(LD)

        # Calculo da profundidade do núcleo [n]
        e = [np.sin(teta[0]) * dc / 2]
        for i, _ in enumerate(teta[1:]):
            x = np.sin(teta[i+1]) * dc / 2 - sum(e)
            e.append(x)

        Abc = np.sum(L * np.asarray(e, np.float64)) * 2

        # é a profundidade total do núcleo do transformador
        # Prof = np.sum(e) * 2

        k = self.tables.constante_tipo_isolacao(tipo, numero_degraus)

        # import ipdb; ipdb.set_trace()
        d = sqrt(Ac / k)

        # import ipdb; ipdb.set_trace()

        # x = (Ac * 4 / pi) ** .5
        # TODO talvez vamos calcular d usando d como o diametro de Ac

        # TODO Kw deveria vir da tabela 2.1.
        # na equação 2.26 usa esse Kw e diz que ele é definido na tabela 2.1
        Kw = Ksw / (30 + Vf2)

        # é a área da janela em [mm²]. Ac esta em [mm²]
        Aw = S / (3.33 * f * Ac * Bm * Kw * Jbt) * 1e9

        # TODO essa equação não está na tese. Olhar a página 50
        ww = sqrt(Aw / Rjan)

        hw = Aw / ww            # é a altura da janela [m]

        # é a maior largura da coluna do núcleo do transformador [m]
        wc = L[0]
        # é a distância entre os centros de duas colunas [m]
        D = ww + wc
        W = 2 * D + wc          # é a largura total do núcleo [m]

        # a = (d - wc) / 2
        # Estimativa da corrente de carga
        # Abj = rel * Abc     # é a área bruta da culatra [mm²]
        Aj = rel * Ac       # é a área do jugo ou da culatra [mm²]
        # hy = Abj / Prof     # é a altura da culatra [mm]
        By = Bm / rel       # densidade de fluxo no jugo (yoke)
        # TODO entender o que é isso

        # H = hw + 2 * hy     # é a altura total do núcleo [m]
        Vferc = 3 * hw * Ac  # Volume de ferro no núcleo [mm³]
        Bfe = Dfe * 1e-9     # Densidade do ferro em [Kg / mm³]
        Mc = Vferc * Bfe     # Massa da culatra  [Kg]

        # import ipdb; ipdb.set_trace()
        # TODO realizar os testes a partir daqui

        Pic = self.tables.perda_magnetica_nucleo(Bm)

        Wic = Pic * Mc          # perda específica no núcleo [W]
        Vferj = Aj * W * 2      # é o volume do ferro nas culatras
        Mj = Vferj * Bfe
        MT = Mj + Mc            # Massa total do Trafo [Kg]
        # import ipdb; ipdb.set_trace()

        Pij = self.tables.perda_magnetica_nucleo(By)

        Wij = Pij * Mj
        # Perdas totais do ferro no transformador.
        # As perdas totais é a soma da perda nas colunas
        # + as perdas nas culatras (culatras e guarnições)
        Po = (Wic + Wij) * 1.05
        # Componente da corrente ativa Ip da perda no núcleo [A]
        # Ip = Po/(3 * Vf1) * 1e-3

        # atc = self.tables.curva_BH(Bm)
        # atj = self.tables.curva_BH(By)
        # from ipdb import set_trace; set_trace()

        # ATj = 2 * W * atj   # A força magnetomotriz na culatra [Ae]
        # ATc = 3 * hw * atc  # A força magnetomotriz na coluna [Ae]

        # ATcj = ATc + ATj        # A força magnetomotriz total [Ae]
        # N1 enrolamento do lado da BT conforme trafo WEG DE 150 KVA
        # Iq = ATcj / N1 * 1e-3

        # Io = sqrt(Ip ** 2 + Iq ** 2) # A corrente a vazio [A]
        # import ipdb; ipdb.set_trace()

        I1 = S / 3 / Vf1   # FIXME S é a potencia total do Trafo
        Fc1 = I1 / Jbt
        Swind1 = Fc1 * N1
        z = (hw * Kw) * 2
        hb = (hw - z) * 1.11
        tbt1 = Swind1 / hb * 1.1     # 10% de folga
        tbt2 = tbt1 * 2

        Dextbt = tbt2 + d           # diametro em mm
        # diâmetro médio na baixa tensão em milímetros
        dmbt = (Dextbt + d) / 2
        Lmbt = pi * dmbt            # Comprimento médio em mm
        Compbt1 = Lmbt * N1         # Comprimento do fio na baixa tensão
        # dfc1 = sqrt(4 * Fc1 / pi)
        VALbt = Compbt1 * Fc1 * 3

        Mbt3 = VALbt * Dal * 1e-9
        # I2menor = (S / 3 / Vf2)     # corrente no secundário
        # TODO pergutar ao professor se esse valor deve ser dado do usuário
        Vmedio2 = 12
        # tap intermediário para dimensionamento dos condutores
        I2c = S / 3 / Vmedio2

        Fc2AT = I2c / Jat           # área do condutor em mm2 no secundário
        # dfc2AT  = sqrt(Fc2AT * 4 / pi)  # Diametro do condutor em mm
        SwindAT = Fc2AT * N2            # Area referente a alta tensao
        # dAt = sqrt(SwindAT * 4 / pi)

        tAT1 = SwindAT / hb * 1.1
        # Laxju = hw - hw * Kw / 2

        # tAT2 = tAT1 * 2

        dintAT = Dextbt + 6 * (d - wc) / 2
        DextAT = dintAT + 4 * tAT1

        LmATc = pi * (dintAT + DextAT) / 2      # Comprimento em milímetros
        CompAT = LmATc * N2

        # dfc2 = sqrt(I2c / Jat * 4 / pi)
        VALAT = CompAT * I2c / Jat * 3
        MAT3 = VALAT * Dal * 1e-9

        # Resistencia do cobre no lado da baixa tesão
        R1 = 0.02857 * Compbt1 / Fc1 * 1e-3
        # Resistencia do cobre no lado da alta tesão
        R2 = 0.02857 * CompAT / Fc2AT * 1e-3

        I2 = S / 3 / Vf2
        Pj = (R1 * I1 ** 2 + R2 * I2 ** 2) * 3
        PerdasT = Po + Pj
        Mativa = MAT3 + Mbt3 + MT
        # import ipdb; ipdb.set_trace()

        return np.array([PerdasT, Mativa], dtype=np.float64)
