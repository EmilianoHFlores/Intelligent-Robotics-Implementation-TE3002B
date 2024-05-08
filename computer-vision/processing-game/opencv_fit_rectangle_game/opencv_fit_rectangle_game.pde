// 
int rectSize = 50;
int floorX, floorY, floorWidth, floorHeight;
float rectAngle = 0;

int floorHoleWidth, floorHoleHeight, floorX, floorY;

void setup() {
  size(800, 600);
  floorX = 0;
  floorY = 2*height / 3;
  floorWidth = width;
  floorHeight = height;
  floorHoleWidth = 60;
  floorHoleHeight = 100;
  floorX = width / 2 - floorHoleWidth / 2;
  floorY = floorY - floorHoleHeight;
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
  // Draw the floor
  fill(50, 150, 200);
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
  if (rectLowestY < floorY || rectLowestX < floorX || rectHighestX > floorX + floorWidth) {
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
  }
}
