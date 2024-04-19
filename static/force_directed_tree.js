// Set up the SVG canvas to fill the window
const svg = d3.select('#tree')
  .attr('width', window.innerWidth)
  .attr('height', window.innerHeight);

const g = svg.append('g');

// Define zoom behavior for the SVG canvas
const zoom = d3.zoom()
  .on('zoom', (event) => {
    g.attr('transform', event.transform);
  });

svg.call(zoom);

// Listen for window resize events and adjust the SVG size
window.addEventListener('resize', () => {
  svg.attr('width', window.innerWidth)
     .attr('height', window.innerHeight);
});

// Declare the simulation variable in a wider scope
let simulation;

// Global reference to the root node
let globalRoot;

// Fetch the MCTS tree data and create visualization
fetch('/tree')
  .then(response => response.json())
  .then(data => {
    globalRoot = d3.hierarchy(data); // Update globalRoot with the fetched data
    createForceDirectedGraph(globalRoot); // Pass the globalRoot for initial graph creation
  });

function createForceDirectedGraph(root) {
  calculateDepth(root); // Calculate the depth for each node
  let nodes = flatten(root);
  let links = root.links();

  // Initialize the simulation with basic forces adjusted for tree layout
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 50 + d.source.depth * 20))
    .force("charge", d3.forceManyBody().strength(-500))
    .force("y", d3.forceY().y(d => d.depth * 100).strength(1))
    .force("x", d3.forceX(window.innerWidth / 2).strength(0.1));

  // Initial positions to avoid undefined coordinates
  nodes.forEach(d => {
    d.x = d.depth * 100 + 50;
    d.y = window.innerHeight / 2;
  });

  simulation.on("tick", () => update(nodes, links));
  update(nodes, links); // Initial update call to render the graph
}

function update(nodes, links) {
  let link = g.selectAll(".link").data(links, d => `${d.source.id}-${d.target.id}`);
  link.exit().remove();
  link = link.enter().append("line")
    .classed("link", true)
    .attr("stroke-width", 2)
    .attr("stroke", "#999")
    .merge(link)
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  let node = g.selectAll(".node").data(nodes, d => d.id);
  node.exit().remove();
  const nodeEnter = node.enter().append("g")
    .classed("node", true)
    .call(drag(simulation));

  nodeEnter.append("circle")
    .attr("r", 5)
    .on('click', (event, d) => {
      toggleChildren(d); // Now this function doesn't need 'root' passed directly
    });

  nodeEnter.append("title")
    .text(d => `Board:\n${formatBoard(d.data.board)}`);

  node = nodeEnter.merge(node)
    .attr("transform", d => `translate(${d.x},${d.y})`);
}

function toggleChildren(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else if (d._children) {
    d.children = d._children;
    d._children = null;
  }
  // Use the globalRoot to recreate the graph
  createForceDirectedGraph(globalRoot);
  simulation.alpha(1).restart(); // Restart simulation to reflect changes
}

function flatten(root) {
  let nodes = [];
  function recurse(node) {
    if (!node.id) node.id = `node-${nodes.length}`;
    nodes.push(node);
    if (node.children) {
      node.children.forEach(recurse);
    }
  }
  recurse(root);
  return nodes;
}

function drag(simulation) {
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  return d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended);
}

function formatBoard(board) {
  return board.map(row => row.join(' ')).join('\n');
}

function calculateDepth(root) {
  function recurse(node, depth = 0) {
    if (!node.depth) node.depth = depth;
    if (node.children) {
      node.children.forEach(child => recurse(child, depth + 1));
    }
  }
  recurse(root);
}
