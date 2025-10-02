###
# Multi-stage build for the Angular front-end
###

# Stage 1: Build the Angular application.  Use the official Node image
# because it contains all tools necessary to run npm and Angular CLI.
FROM node:18-alpine as build

WORKDIR /usr/src/app

# Install global Angular CLI.  While optional, it ensures the correct
# version of the CLI is available inside the container to build the
# project.  npm ci is used if package-lock.json is present for faster
# installs.
RUN npm install -g @angular/cli@16

COPY package*.json ./
RUN npm ci

# Copy application code and run the build.  The build command writes
# compiled assets to the dist/ directory.  By default Angular outputs to
# dist/<project-name>.  To simplify copy instructions, we leave the
# directory name unspecified and copy whatever exists in dist/ to Nginx
# later.
COPY . .
RUN ng build --configuration production

# Stage 2: Serve the built application with Nginx.  Use the small
# official Nginx image as the final runtime environment.
FROM nginx:1.25-alpine

# Remove the default Nginx site configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built Angular app from the build stage.  This copies all
# files from the dist/ directory into Nginx's html root.  The wildcard
# dist/* works regardless of the Angular project name.
COPY --from=build /usr/src/app/dist/ /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
