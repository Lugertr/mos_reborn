# ====== Build stage ======
FROM node:20-alpine AS build
WORKDIR /app

# ставим зависимости
COPY package*.json ./
RUN npm ci

# исходники
COPY . .

# Angular 19 по умолчанию кладёт браузерный билд в dist/<project>/browser.
# Явно фиксируем выходной путь, чтобы было однозначно куда копировать.
RUN npx ng build --configuration=production

# ====== Runtime (nginx) ======
FROM nginx:1.27-alpine AS runtime

# наш главный конфиг (без include conf.d)
COPY nginx.conf /etc/nginx/nginx.conf

# билд в web-root образа
COPY --from=build /app/dist/frontend/browser/ /usr/share/nginx/html/

EXPOSE 4200
CMD ["nginx", "-g", "daemon off;"]
