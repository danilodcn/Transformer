from utils.memory import Memory
from numpy import interp


class Tables:

    tables = Memory()

    def __init__(self, tables: dict):
        self.tables.set_many(tables)

    def get_table(self, table_name):
        tabela = self.tables.get(table_name).get("dados")
        return tabela

    def numero_degraus(self, area: float):
        tabela = self.get_table("numero_degraus")
        # from ipdb import set_trace; set_trace()

        if area >= 200 or area <= 0:
            raise ValueError("A a area nao pode ser maior 0.2 m² nem menor 0")

        # import ipdb; ipdb.set_trace()
        if area < tabela.get("1")[0]:
            return 1

        # from ipdb import set_trace; set_trace()

        for key, value in list(tabela.items())[1:]:
            i, j = value
            if i <= area < j:
                return int(key)

        # raise ValueError("Houve um erro!!")

    def dimensoes_nucleo(self, numero_degraus: int):
        numero_degraus -= 1
        tabela = self.get_table("dimensoes_nucleo")

        # from ipdb import set_trace; set_trace()
        Ku = tabela["Ku"][numero_degraus]
        L = tabela["dimensoes_nucleo"][numero_degraus]

        return Ku, L

    def constante_tipo_isolacao(self, tipo, numero_degraus):
        tabela = self.get_table("constante_tipo_isolacao")

        numero_degraus -= 1
        if numero_degraus > 4:
            numero_degraus = 4
        try:
            return tabela[tipo][numero_degraus]

        except Exception:
            txt = f'O tipo de transformador "{tipo}" não é suportado.'

            msg = 'Os tipos suportados sao: [{}]'
            msg.format(', '.join([f'"{i}"' for i in tabela.keys()]))
            txt += msg
            raise KeyError(txt)

    def perda_magnetica_nucleo(self, Bm):
        tabela = self.get_table("perda_magnetica")

        return interp(
            Bm,
            tabela.get("inducao"),
            tabela.get("perdas")
        )

    def curva_BH(self, B):
        tabela = self.get_table("curva_BH")
        # from ipdb import set_trace; set_trace()

        return interp(
            B,
            tabela.get("B"),
            tabela.get("H")
        )
