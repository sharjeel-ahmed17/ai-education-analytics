// ==========================================
// CORE APP STATE & EVENTS
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initSidebar();
    initUploadZone();
    initReportsForm();
});

// ==========================================
// THEME SWITCHING (LIGHT/DARK)
// ==========================================
function initTheme() {
    const themeToggleBtn = document.getElementById("theme-toggle");
    const themeIcon = document.getElementById("theme-icon");
    const htmlEl = document.documentElement;

    // Load theme preference
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlEl.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const currentTheme = htmlEl.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        
        htmlEl.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
        updateThemeIcon(newTheme);
    });
}

function updateThemeIcon(theme) {
    const themeIcon = document.getElementById("theme-icon");
    if (theme === "dark") {
        themeIcon.className = "bx bx-sun";
    } else {
        themeIcon.className = "bx bx-moon";
    }
}

// ==========================================
// SIDEBAR COLLAPSE & ROUTING
// ==========================================
function initSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggleBtn = document.getElementById("sidebar-toggle");
    const menuLinks = document.querySelectorAll(".sidebar-link");

    // Toggle Collapse
    toggleBtn.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
    });

    // Client-side Routing / Section Toggling
    menuLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const targetSectionId = link.getAttribute("data-section");
            
            // Update Active Menu
            menuLinks.forEach(l => l.classList.remove("active"));
            link.classList.add("active");

            // Update Active Section
            switchSection(targetSectionId);
        });
    });
}

function switchSection(sectionId) {
    const sections = document.querySelectorAll(".app-section");
    sections.forEach(sec => {
        sec.classList.remove("active");
        if (sec.id === sectionId) {
            sec.classList.add("active");
        }
    });

    // Highlight sidebar active link if triggered from external buttons
    const menuLinks = document.querySelectorAll(".sidebar-link");
    menuLinks.forEach(link => {
        if (link.getAttribute("data-section") === sectionId) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
}

// ==========================================
// FILE UPLOAD SYSTEM
// ==========================================
let uploadedFiles = [];

function initUploadZone() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileListContainer = document.getElementById("file-list-container");
    const selectedFilesDiv = document.getElementById("selected-files");
    const processBtn = document.getElementById("btn-process-upload");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener("change", () => {
        handleFiles(fileInput.files);
    });

    function handleFiles(files) {
        for (let i = 0; i < files.length; i++) {
            uploadedFiles.push(files[i]);
        }
        updateFileList();
    }

    function updateFileList() {
        if (uploadedFiles.length > 0) {
            fileListContainer.style.display = "flex";
            selectedFilesDiv.innerHTML = "";
            uploadedFiles.forEach((file, index) => {
                const item = document.createElement("div");
                item.style.display = "flex";
                item.style.justifyContent = "space-between";
                item.style.alignItems = "center";
                item.style.padding = "8px 12px";
                item.style.backgroundColor = "var(--bg-card)";
                item.style.border = "1px solid var(--border-color)";
                item.style.borderRadius = "6px";
                
                let fileIcon = "bx bx-file";
                if (file.name.endsWith(".csv")) fileIcon = "bx bx-spreadsheet";
                if (file.name.endsWith(".pdf")) fileIcon = "bx bxs-file-pdf";
                
                item.innerHTML = `
                    <div style="display:flex; align-items:center; gap:8px; font-size:0.85rem;">
                        <i class="${fileIcon}" style="font-size:1.1rem; color:var(--color-primary);"></i>
                        <span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                    <button class="btn btn-secondary btn-sm" style="padding:2px 6px; font-size:0.75rem;" onclick="removeUploadedFile(${index})">
                        <i class="bx bx-trash" style="color:var(--color-danger)"></i>
                    </button>
                `;
                selectedFilesDiv.appendChild(item);
            });
        } else {
            fileListContainer.style.display = "none";
        }
    }

    window.removeUploadedFile = (index) => {
        uploadedFiles.splice(index, 1);
        updateFileList();
    };

    processBtn.addEventListener("click", () => {
        processBtn.innerHTML = `<i class="bx bx-loader-alt bx-spin"></i> Processing files...`;
        processBtn.disabled = true;

        setTimeout(() => {
            alert(`Successfully ingested and processed ${uploadedFiles.length} file(s) into database!`);
            processBtn.innerHTML = `<i class="bx bx-cpu"></i> Ingest & Preprocess Files`;
            processBtn.disabled = false;
            uploadedFiles = [];
            updateFileList();
            switchSection("predictions");
        }, 1500);
    });
}

// ==========================================
// EXPLAINABILITY (XAI) VISUALIZER
// ==========================================
function viewExplainability(name, score, shapValues) {
    switchSection("explainability");
    
    // Update labels and values
    document.getElementById("xai-student-title").innerText = `Selected Student: ${name}`;
    document.getElementById("xai-risk-badge").innerText = `${score}% Risk Score`;

    // Map factors to element keys
    const factors = {
        att: shapValues.Attendance,
        fail: shapValues.Failures,
        g1: shapValues.G1,
        g2: shapValues.G2,
        study: shapValues.StudyTime
    };

    for (const [key, val] of Object.entries(factors)) {
        const fillEl = document.getElementById(`bar-${key}`);
        const valEl = document.getElementById(`val-${key}`);
        
        // Render percentage width
        const absVal = Math.abs(val);
        const percent = Math.min(Math.round(absVal * 200), 100); // Scale multiplier for visualization
        
        fillEl.style.width = `${percent}%`;
        
        // Style values directions
        if (val >= 0) {
            valEl.innerText = `+${val.toFixed(2)}`;
            valEl.className = "xai-bar-val positive";
            fillEl.className = "progress-bar-fill danger";
            
            // Fix text-direction for study times
            if (key === 'study') {
                fillEl.parentElement.style.direction = "ltr";
            }
        } else {
            valEl.innerText = `${val.toFixed(2)}`;
            valEl.className = "xai-bar-val negative";
            fillEl.className = "progress-bar-fill success";
            
            if (key === 'study' || key === 'att' || key === 'fail' || key === 'g1' || key === 'g2') {
                fillEl.parentElement.style.direction = "rtl";
            }
        }
    }
}

// ==========================================
// HUMAN IN THE LOOP (HITL) WORKFLOW
// ==========================================
let pendingHITLCount = 2;
let approvedHITLCount = 3;

function queueReview(id, name, score, reason) {
    const container = document.getElementById("hitl-queue-container");
    const itemIndex = Date.now();
    
    const newItem = document.createElement("div");
    newItem.className = "card hitl-card";
    newItem.id = `hitl-item-${itemIndex}`;
    
    newItem.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
            <div>
                <h3 style="font-size: 1.15rem; font-weight: 700;">${name} (ID: ${id})</h3>
                <p style="font-size: 0.85rem; color: var(--text-muted);">Predictive Risk: <strong style="color: var(--color-danger);">${score}%</strong> &bull; Class Grade Average: <b>C</b></p>
            </div>
            <span class="badge-tag danger">Awaiting Audit</span>
        </div>
        
        <div style="background-color: var(--bg-primary); padding: 14px; border-radius: 8px; border-left: 3px solid var(--color-primary); font-size: 0.9rem; margin-bottom: 15px;">
            <strong style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 4px;">LLM Synthesized RAG Context:</strong>
            "${reason}"
        </div>

        <div class="form-group">
            <label class="form-label">Reviewer Comments & Adjustments</label>
            <textarea class="form-control" rows="2" placeholder="Modify RAG action details or write custom recommendation instructions..."></textarea>
        </div>

        <div style="display: flex; justify-content: flex-end; gap: 10px;">
            <button class="btn btn-secondary" onclick="rejectHITL('hitl-item-${itemIndex}')">Reject Alert</button>
            <button class="btn btn-primary" onclick="approveHITL('hitl-item-${itemIndex}', '${name}')">Approve Report</button>
        </div>
    `;
    
    container.insertBefore(newItem, container.firstChild);
    
    pendingHITLCount++;
    updateHITLBadges();
    alert(`${name} successfully added to Human-in-the-Loop review queue.`);
}

function approveHITL(itemId, studentName) {
    const item = document.getElementById(itemId);
    if (!item) return;

    item.style.transform = "scale(0.95)";
    item.style.opacity = "0";
    
    setTimeout(() => {
        item.remove();
        pendingHITLCount = Math.max(0, pendingHITLCount - 1);
        approvedHITLCount++;
        updateHITLBadges();
        
        // Add to reports section dropdown options and list
        const reportsList = document.getElementById("reports-list");
        const reportIndex = Date.now();
        const reportItem = document.createElement("div");
        reportItem.style = "display: flex; justify-content: space-between; align-items: center; padding: 12px; background-color: var(--bg-primary); border-radius: 8px; border-left: 3px solid var(--color-success);";
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        reportItem.innerHTML = `
            <div>
                <h4 style="font-size: 0.85rem; font-weight: 700;">${studentName} Report</h4>
                <p style="font-size: 0.75rem; color: var(--text-muted);">PDF Generated &bull; Today, ${timeStr}</p>
            </div>
            <button class="btn btn-secondary btn-sm" style="padding: 4px 8px; font-size: 0.75rem;"><i class="bx bx-download"></i></button>
        `;
        reportsList.insertBefore(reportItem, reportsList.firstChild);

        alert(`Intervention report approved and compiled for ${studentName}! Added to Reports & Exports page.`);
    }, 300);
}

function rejectHITL(itemId) {
    const item = document.getElementById(itemId);
    if (!item) return;

    item.style.transform = "scale(0.95)";
    item.style.opacity = "0";
    
    setTimeout(() => {
        item.remove();
        pendingHITLCount = Math.max(0, pendingHITLCount - 1);
        updateHITLBadges();
        alert(`Risk alert removed from review queue.`);
    }, 300);
}

function updateHITLBadges() {
    document.getElementById("hitl-badge-count").innerText = pendingHITLCount;
    document.getElementById("pending-reviews-count").innerText = pendingHITLCount;
    document.getElementById("hitl-pending-count").innerText = pendingHITLCount;
    document.getElementById("hitl-approved-count").innerText = approvedHITLCount;
}

// ==========================================
// REPORTS & INTERVENTIONS FORM
// ==========================================
function initReportsForm() {
    window.generateReport = () => {
        const student = document.getElementById("report-student").value;
        
        alert(`Generating report for ${student}...`);
        
        setTimeout(() => {
            const reportsList = document.getElementById("reports-list");
            const reportItem = document.createElement("div");
            reportItem.style = "display: flex; justify-content: space-between; align-items: center; padding: 12px; background-color: var(--bg-primary); border-radius: 8px; border-left: 3px solid var(--color-success);";
            
            const now = new Date();
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            reportItem.innerHTML = `
                <div>
                    <h4 style="font-size: 0.85rem; font-weight: 700;">${student} Report</h4>
                    <p style="font-size: 0.75rem; color: var(--text-muted);">PDF Generated &bull; Today, ${timeStr}</p>
                </div>
                <button class="btn btn-secondary btn-sm" style="padding: 4px 8px; font-size: 0.75rem;"><i class="bx bx-download"></i></button>
            `;
            reportsList.insertBefore(reportItem, reportsList.firstChild);
            alert(`Successfully compiled intervention plan for ${student}! PDF report downloaded.`);
        }, 1000);
    };
}
