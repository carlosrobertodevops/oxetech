# Agent Instructions - AV1 Projeto de Frontend

Este repositório contém a Avaliação 1 (AV1) da disciplina de Desenvolvimento Web Full Stack (DWFS).

## Escopo do Projeto
Trata-se de uma aplicação frontend (Gerenciador de Tarefas) utilizando apenas:
- HTML Semântico
- CSS (Tailwind via CDN)
- JavaScript Vanilla (Manipulação de DOM)

## Regras de Atuação
- **Zero Build Tools:** Não introduza NPM, Node.js, empacotadores (Vite/Webpack) ou frameworks reativos (React, Vue).
- **Apenas Frontend:** Todo o código deve rodar nativamente no browser.
- **Estrutura Esperada:**
  - `index.html` (com importação do Tailwind)
  - `css/style.css`
  - `js/app.js`
  - `README.md`
- Ao fazer modificações em estilos, prefira classes utilitárias do Tailwind no arquivo HTML. Use o `style.css` apenas para necessidades estritas não cobertas pelo framework.
- Ao fazer alterações no JS, siga a lógica orientada a eventos (`addEventListener`) e manipulação direta do DOM (`document.createElement`, `appendChild`).