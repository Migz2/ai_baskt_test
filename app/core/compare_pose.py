def gerar_dicas(user_kpts, ref_kpts):
    dicas = []
    # Simples placeholder baseado em diferenças visuais (pode ser aprimorado)
    if len(user_kpts) == 0 or len(ref_kpts) == 0:
        return ["Não foi possível analisar os movimentos."]

    try:
        if user_kpts[0][13][0] - user_kpts[0][15][0] > 0.1:
            dicas.append("Seu cotovelo está muito aberto antes do arremesso.")
        if user_kpts[0][11][1] - user_kpts[0][23][1] > 0.1:
            dicas.append("Seu tronco está inclinando para trás na hora do arremesso.")
        if user_kpts[5][13][1] < user_kpts[5][23][1]:
            dicas.append("Seu braço está subindo antes da perna impulsionar.")
    except:
        dicas.append("Dificuldade ao gerar dicas precisas com os dados fornecidos.")

    return dicas