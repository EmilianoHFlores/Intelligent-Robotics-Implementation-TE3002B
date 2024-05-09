import processing.net.*; 
Client client;
int port = 5204;
String ip = "127.0.0.1";
boolean VERBOSE = false;

int rectSize = 50;
int floorX, floorY, floorWidth, floorHeight;
int rectX = 0; int rectY = 0;
float rectAngle = 0;

int floorHoleWidth, floorHoleHeight, floorHoleX, floorHoleY;


// IMAGE SIZE
int imgWidth = 800;
int imgHeight = 600;

// a matrix where the collision points are stored
boolean[][] collisionPoints = new boolean[imgWidth][imgHeight];
boolean[][] rectCollisionPoints = new boolean[imgWidth][imgHeight];
boolean collision;

// Equations of the 4 sides of the rectangle, where the first element is the slope and the second is the y-intercept
float rectEquations[][] = {
  {1, 0},
  {0, 1},
  {1, 1},
  {-1, 1}
};

void setup() {
  size(800, 600);
  floorX = 0;
  floorY = 2*height / 3;
  floorWidth = width;
  floorHeight = height;
  floorHoleWidth = 60;
  floorHoleHeight = 100;
  floorHoleX = width / 2 - floorHoleWidth / 2;
  floorHoleY = floorY + floorHoleHeight;
  print("floorHoleY: " + floorHoleY + "\n");
  print("floorY: " + floorY + "\n");
  // set the collision points according to the floor
  for (int i = 0; i < imgWidth; i++) {
    for (int j = 0; j < imgHeight; j++) {
      if (j < floorY || (j < floorHoleY && i > floorHoleX && i < floorHoleX + floorHoleWidth)) {
        collisionPoints[i][j] = false;
      } else {
        collisionPoints[i][j] = true;
      }
    }
  }
  for (int i = 0; i < imgWidth; i++) {
    for (int j = 0; j < imgHeight; j++) {
      rectCollisionPoints[i][j] = false;
    }
  }

  // Connect to the server's IP address and port
  client = new Client(this, ip, port);
  
}

// if keyboard is pressed, rotate the rectangle (+ or -)
void keyPressed() {
  print(key);
  if (key == 'a') {
    rectAngle += PI/10;
  } else if (key == 'b') {
    rectAngle -= PI/10;
  }
}

void draw() {
  background(100, 200, 255);

  // if client receives a message, print it
  if (client.available() > 0) {
    String message = client.readString();
    print(message);
    print("\n");
    // message comes in format "x,y,angle"
    String[] parts = split(message, ",");
    rectX = int(parts[0]);
    rectY = int(parts[1]);
    rectAngle = float(parts[2]);
  }
  // draw collision points
  // fill(0, 0, 0);
  // for (int i = 0; i < imgWidth; i++) {
  //  for (int j = 0; j < imgHeight; j++) {
  //    if (collisionPoints[i][j]) {
  //      point(i, j);
  //    }
  //  }
  //  point(i, floorHoleY);
  // }
  // Draw the floor
  /*fill(50, 150, 200);
  rectMode(CORNER);
  rect(floorX, floorY, floorWidth, floorHeight);

  // Draw the hole in the floor
  fill(100, 200, 255); // Light blue
  rect(floorWidth / 2 - floorHoleWidth / 2, floorY, floorHoleWidth, floorHoleHeight);

  // get rectangle lowest point, considering rotation
  float rectLowestY = mouseY + (rectSize / 2) * cos(rectAngle) + (rectSize / 2) * sin(rectAngle);
  float rectLowestX = mouseX + (rectSize / 2) * cos(rectAngle) - (rectSize / 2) * sin(rectAngle);
  float rectHighestX = mouseX - (rectSize / 2) * cos(rectAngle) + (rectSize / 2) * sin(rectAngle);
  
  
  // Check if the mouse rectangle is inside the floor
  if (rectLowestY < floorY || rectLowestX < floorHoleX || rectHighestX > floorHoleX + floorWidth) {
        // Draw the rectangle at the mouse position
        fill(255, 255, 255); // Green
        noStroke();
        rectMode(CENTER);
        pushMatrix();
        translate(mouseX, mouseY);
        rotate(rectAngle);
        rect(0, 0, rectSize, rectSize);
        popMatrix();
    // Mouse rectangle is outside the floor, set background to red
  } else {
    // Mouse rectangle is inside the floor, set background to blue
    // Draw the rectangle at the mouse position
    fill(255, 0, 0); // Green
    noStroke();
    rectMode(CENTER);
    // draw rotated 45 degrees
    pushMatrix();
    translate(mouseX, mouseY);
    rotate(rectAngle);
    rect(0, 0, rectSize, rectSize);
    popMatrix();
  }*/

  // Draw the floor
  fill(50, 150, 200);
  // no outline
  noStroke();
  rectMode(CORNER);
  rect(floorX, floorY, floorWidth, floorHeight);
  
  // Draw the hole in the floor
  fill(100, 200, 255); // Light blue
  rect(floorHoleX, floorY, floorHoleWidth, floorHoleHeight);
  // get rectangle collision points, considering rotation
  for (int i = 0; i < imgWidth; i++) {
    for (int j = 0; j < imgHeight; j++) {
      rectCollisionPoints[i][j] = false;
    }
  }
  for (int i = 0; i < rectSize; i++) {
    for (int j = 0; j < rectSize; j++) {
      int x = rectX - rectSize / 2 + i;
      int y = rectY - rectSize / 2 + j;
      int xRotated = (int) (rectX + (i - rectSize / 2) * cos(rectAngle) - (j - rectSize / 2) * sin(rectAngle));
      int yRotated = (int) (rectY + (i - rectSize / 2) * sin(rectAngle) + (j - rectSize / 2) * cos(rectAngle));
      if (xRotated >= 0 && xRotated < imgWidth && yRotated >= 0 && yRotated < imgHeight) {
        rectCollisionPoints[xRotated][yRotated] = true;
      }
    }
  }
  // Draw the rectangle collisions
  collision = false;
  for (int i = 0; i < imgWidth; i++) {
    for (int j = 0; j < imgHeight; j++) {
      if (rectCollisionPoints[i][j] && collisionPoints[i][j]) {
        stroke(255, 0, 0);
        point(i, j);
        collision = true;
      }
      else if (rectCollisionPoints[i][j]) {
        stroke(0, 255, 0);
        point(i, j);
      }
    }
  }
  if (!VERBOSE){
    fill(255, 255, collision ? 0 : 255); // Green
    noStroke();
    rectMode(CENTER);
    pushMatrix();
    translate(rectX, rectY);
    rotate(rectAngle);
    rect(0, 0, rectSize, rectSize);
    popMatrix();
  }
}
