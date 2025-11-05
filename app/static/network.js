document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById("network-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Set canvas size to left half of viewport
    function resizeCanvas() {
        canvas.width = window.innerWidth / 2;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Network animation config (vibrant, modern)
    const config = {
        numberOfNodes: 40,
        maxDistance: 150,
        nodeRadius: 4,
        speed: 0.3,
        nodeColor: "#00FFF0",
        nodeGlow: "rgba(0,255,240,0.6)",
        lineColor: "rgba(0,217,204,0.4)",
        lineGlowColor: "rgba(0,255,240,0.2)",
        lineWidth: 2,
        nodeShadowBlur: 15,
        nodeShadowColor: "#00FFF0",
        spawnArea: {
            minX: 50,
            maxX: () => canvas.width - 50,
            minY: 50,
            maxY: () => canvas.height - 50
        }
    };

    function createNode() {
        return {
            x: Math.random() * (config.spawnArea.maxX() - config.spawnArea.minX) + config.spawnArea.minX,
            y: Math.random() * (config.spawnArea.maxY() - config.spawnArea.minY) + config.spawnArea.minY,
            vx: (Math.random() - 0.5) * config.speed,
            vy: (Math.random() - 0.5) * config.speed
        };
    }

    let nodes = [];
    for (let i = 0; i < config.numberOfNodes; i++) {
        nodes.push(createNode());
    }

    function updateNodes() {
        for (let node of nodes) {
            node.x += node.vx;
            node.y += node.vy;
            // Bounce off edges
            if (node.x < config.spawnArea.minX || node.x > config.spawnArea.maxX()) node.vx *= -1;
            if (node.y < config.spawnArea.minY || node.y > config.spawnArea.maxY()) node.vy *= -1;
            node.x = Math.max(config.spawnArea.minX, Math.min(node.x, config.spawnArea.maxX()));
            node.y = Math.max(config.spawnArea.minY, Math.min(node.y, config.spawnArea.maxY()));
        }
    }

    function drawConnections() {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < config.maxDistance) {
                    const opacity = 1 - (distance / config.maxDistance);
                    ctx.save();
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = config.lineGlowColor;
                    ctx.strokeStyle = `rgba(0,217,204,${opacity * 0.7})`;
                    ctx.lineWidth = config.lineWidth;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }
    }

    function drawNodes() {
        for (let node of nodes) {
            ctx.save();
            ctx.shadowBlur = config.nodeShadowBlur;
            ctx.shadowColor = config.nodeShadowColor;
            ctx.fillStyle = config.nodeColor;
            ctx.beginPath();
            ctx.arc(node.x, node.y, config.nodeRadius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore();
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawConnections();
        drawNodes();
        updateNodes();
        requestAnimationFrame(animate);
    }

    animate();
});
