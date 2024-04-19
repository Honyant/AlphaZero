// Set up the SVG canvas to fill the window
const svg = d3.select('#tree')
  .attr('width', window.innerWidth)
  .attr('height', window.innerHeight)
  .style("user-select", "none");

const margin = {top: 20, right: 120, bottom: 20, left: 40};
const width = +svg.attr("width");
const dy = width / 8;
const dx = 50; // Vertical space between the nodes
const g = svg.append('g').attr("transform", `translate(${margin.left},${margin.top})`);

// Define zoom behavior for the SVG canvas
const zoom = d3.zoom()
  .on('zoom', (event) => {
    g.attr('transform', event.transform);
  });

svg.call(zoom);

// Tooltip for Connect4 board visualization
const tooltip = svg.append("g")
  .attr("class", "tooltip")
  .style("display", "none");

// Loading the tree data
fetch('/tree')
  .then(response => response.json())
  .then(data => {
    const root = d3.hierarchy(data, d => {
      // Sort children by actions if they exist
      if (d.children) {
        d.children.sort((a, b) => a.actions[0] - b.actions[0]);
      }
      return d.children;
    });
    root.x0 = dy / 2;
    root.y0 = 0;
    // Collapse all nodes initially except for the root and its immediate children
    // root.children.forEach(collapse);
    createCollapsibleTree(root);
  });

function collapse(d) {
  if (d.children) {
    d.children.forEach(collapse);
    d._children = d.children;
    d.children = null;
  }
}

function createCollapsibleTree(root) {
  const tree = d3.tree().nodeSize([dx, dy]);
  const diagonal = d3.linkVertical().x(d => d.x).y(d => d.y);

  function update(source) {
    const nodes = root.descendants().reverse();
    const links = root.links();

    // Compute the new tree layout.
    tree(root);

    g.selectAll(".link")
      .data(links, d => d.target.id)
      .join("path")
      .attr("class", "link")
      .attr("d", diagonal)
      .attr("fill", "none")
      .attr("stroke", "#555")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", 1.5);

    const node = g.selectAll(".node")
      .data(nodes, d => d.id)
      .join(enter => enter.append("g")
        .attr("class", "node"))
      .attr("transform", d => `translate(${d.x},${d.y})`)
      .on("click", (event, d) => {
        if (event.ctrlKey) {
          // CTRL+click: Expand/Collapse all descendants
          toggleChildren(d, true); // The second parameter is to indicate recursive toggle
        } else {
          // Normal click: Toggle only the immediate children
          toggleImmediateChildren(d);
        }
        update(d);
      });

    node.append("circle")
      .attr("r", 14)
      .attr("fill", d => d.data.cur_player === 1 ? "#6495ED" : "#F08080") // Soft blue and soft red
      .attr("stroke", d => d.data.leaf ? "black" : "#2E8B57") // Black for leaf, darker green for nodes with children
      .attr("stroke-width", 3)
      .attr("cursor", "pointer")
      .on("mouseover", (event, d) => displayBoard(event, d.data))
      .on("mouseout", () => tooltip.style("display", "none"));

    function displayBoard(event, data) {
      const cellSize = 50; // Size of each cell on the board
      const boardWidth = cellSize * data.board[0].length + 10; // Additional padding
      const boardHeight = cellSize * data.board.length + 60; // Additional padding for statistics

      tooltip.style("display", null)
        .attr("transform", `translate(${event.pageX + 10},${event.pageY + 10})`)
        .raise(); // Ensure tooltip is on top

      tooltip.selectAll("*").remove(); // Clear previous board

      // Add a background rectangle
      tooltip.append("rect")
        .attr("width", boardWidth)
        .attr("height", boardHeight)
        .attr("fill", "lightgrey") // Background for visibility
        .attr("stroke", "black")
        .attr("stroke-width", 2);

      // Draw the board pieces and statistics
      data.board.forEach((row, rowIndex) => {
        row.forEach((cell, colIndex) => {
          let fillColor = "white"; // Default for empty cell
          if (cell === 1) fillColor = data.cur_player === 1 ? "blue" : "red"; // Use curPlayer's color
          else if (cell === -1) fillColor = data.cur_player === 1 ? "red" : "blue"; // Use opposite color

          tooltip.append("circle")
            .attr("cx", colIndex * cellSize + cellSize / 2 + 5) // Centered + padding
            .attr("cy", rowIndex * cellSize + cellSize / 2 + 5) // Centered + padding
            .attr("r", cellSize / 2 * 0.8) // Slightly smaller than half cellSize for spacing
            .attr("fill", fillColor)
            .attr("stroke", "black");

          // Display the statistics below each column
          if (rowIndex === data.board.length - 1) {
            // Find the index of the current column in the actions array
            const actionIndex = data.actions.indexOf(colIndex);
            if (actionIndex !== -1) {
              // Display the statistics for this column
              const statsY = boardHeight - 40;
              const textX = colIndex * cellSize + cellSize / 2 + 5;
              const stats = [
                `P: ${data.priors[actionIndex].toFixed(2)}`,
                `V: ${data.visits[actionIndex].toFixed(2)}`,
                `Q: ${data.q_values[actionIndex].toFixed(2)}`
              ];
              stats.forEach((stat, i) => {
                tooltip.append("text")
                  .attr("x", textX)
                  .attr("y", statsY + i * 15)
                  .attr("text-anchor", "middle")
                  .text(stat)
                  .attr("font-family", "sans-serif")
                  .attr("font-size", "12px")
                  .attr("fill", "black");
              });
            }
          }
        });
      });
    }
  }

  function toggleChildren(d, recursive) {
    if (recursive) {
      if (d.children) {
        d.children.forEach(child => toggleChildren(child, true));
        d._children = d.children;
        d.children = null;
      } else if (d._children) {
        d._children.forEach(child => toggleChildren(child, true));
        d.children = d._children;
        d._children = null;
      }
    } else {
      toggleImmediateChildren(d);
    }
  }

  function toggleImmediateChildren(d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else if (d._children) {
      d.children = d._children;
      d._children = null;
    }
  }

  update(root);
}
