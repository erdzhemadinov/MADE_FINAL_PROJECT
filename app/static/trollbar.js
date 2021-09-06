var canvas = document.getElementById("myCanvas"), 
context = canvas.getContext("2d");
let colors = ["#29d490", "#98cd6c", "#ffcc33", "#f6964a", "#ee6f62"];
let delta_x = 70;
let delta_y = 10;
for (let i = 0; i < colors.length; i++) {
	context.fillStyle = colors[i];
	context.fillRect(i*delta_x, 5, delta_x, delta_y);
  }