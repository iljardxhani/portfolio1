(() => {
  const yearEl = document.getElementById("year");
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }

  const rootEl = document.documentElement;
  const headerEl = document.querySelector(".site-header");
  const syncHeaderHeight = () => {
    if (!headerEl) {
      return;
    }
    const headerHeight = Math.ceil(headerEl.getBoundingClientRect().height);
    rootEl.style.setProperty("--header-h", `${headerHeight}px`);
  };

  syncHeaderHeight();
  window.addEventListener("resize", syncHeaderHeight, { passive: true });
  window.addEventListener("load", syncHeaderHeight);

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvas = document.getElementById("mesh-canvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  if (canvas && ctx) {

  let w = 0;
  let h = 0;
  let dpr = 1;
  let rafId = null;
  let scrollY = 0;
  let pointerX = 0;
  let pointerY = 0;
  let smoothPointerX = 0;
  let smoothPointerY = 0;
  let pointerVX = 0;
  let pointerVY = 0;

  const mix = (a, b, t) => a + (b - a) * t;

  const resize = () => {
    dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const drawMesh = (timeMs = 0) => {
    const time = timeMs * 0.001;
    const horizon = h * 0.33;
    const rows = Math.max(32, Math.floor(h / 22));
    const cols = Math.max(36, Math.floor(w / 36));
    const phaseScroll = scrollY * 0.001;

    smoothPointerX += (pointerX - smoothPointerX) * 0.085;
    smoothPointerY += (pointerY - smoothPointerY) * 0.085;
    pointerVX *= 0.92;
    pointerVY *= 0.92;

    const pointerSpeed = Math.min(Math.hypot(pointerVX, pointerVY) * 8, 1.4);
    const mu = smoothPointerX * 0.82;
    const mv = clamp((smoothPointerY + 1) * 0.5, 0, 1);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#02060b";
    ctx.fillRect(0, 0, w, h);

    const points = Array.from({ length: rows + 1 }, () => new Array(cols + 1));

    for (let r = 0; r <= rows; r += 1) {
      const t = r / rows;
      const tPow = Math.pow(t, 1.58);
      const yBase = mix(horizon, h * 1.08, tPow);
      const rowWidth = mix(w * 0.18, w * 2.7, Math.pow(t, 1.34));
      const waveAmp = 8 + 44 * Math.exp(-Math.pow((t - 0.34) * 3.6, 2));

      for (let c = 0; c <= cols; c += 1) {
        const u = c / cols - 0.5;

        const waveA =
          Math.sin(u * 10.2 + t * 9.1 + time * 1.12 + phaseScroll) * waveAmp;
        const waveB =
          Math.cos(u * 4.7 - t * 6.4 - time * 0.63 - phaseScroll * 0.64) *
          (waveAmp * 0.46);
        const waveC =
          Math.sin((u + t * 0.7) * 5.6 + time * 0.37) * (waveAmp * 0.22);

        const du = u - mu;
        const dv = t - mv * 0.92;
        const dist2 = du * du * 1.35 + dv * dv * 2.1 + 0.018;
        const influence = Math.min(0.44, 0.028 / dist2);

        const swirlX = -dv * influence * (96 + pointerSpeed * 30);
        const lift = influence * (78 + pointerSpeed * 22) * (1 - t * 0.22);
        const tiltPush = du * influence * (18 + pointerSpeed * 20);

        const x = w * 0.5 + u * rowWidth + swirlX + smoothPointerX * 26 * (1 - t);
        const y =
          yBase +
          waveA +
          waveB +
          waveC -
          lift +
          tiltPush +
          smoothPointerY * 8 * (0.65 - t);

        points[r][c] = { x, y, t, influence };
      }
    }

    for (let r = 0; r <= rows; r += 1) {
      const t = r / rows;
      const alpha = 0.08 + (1 - t) * 0.24;
      ctx.strokeStyle = `rgba(174, 226, 255, ${alpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = 0; c <= cols; c += 1) {
        const point = points[r][c];
        if (c === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      }
      ctx.stroke();
    }

    for (let c = 0; c <= cols; c += 1) {
      const colAlpha = 0.06 + Math.sin((c / cols) * Math.PI) * 0.07;
      ctx.strokeStyle = `rgba(138, 188, 235, ${colAlpha})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let r = 0; r <= rows; r += 1) {
        const point = points[r][c];
        if (r === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      }
      ctx.stroke();
    }

    for (let r = 2; r <= rows; r += 2) {
      for (let c = 1; c <= cols; c += 2) {
        const point = points[r][c];
        const size = 0.62 + point.influence * 2.8;
        const alpha = Math.min(0.86, 0.14 + point.influence * 0.54 + (1 - point.t) * 0.14);
        ctx.fillStyle = `rgba(18,212,255,${alpha})`;
        ctx.beginPath();
        ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const pointerScreenX = w * (0.5 + mu * 0.36);
    const pointerScreenY = mix(horizon + 40, h * 0.82, mv);
    const glow = ctx.createRadialGradient(
      pointerScreenX,
      pointerScreenY,
      0,
      pointerScreenX,
      pointerScreenY,
      280
    );
    glow.addColorStop(0, "rgba(18,212,255,0.2)");
    glow.addColorStop(0.42, "rgba(37,227,180,0.08)");
    glow.addColorStop(1, "rgba(18,212,255,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);

    const ribbonY = horizon + 34 + Math.sin(time * 0.9) * 7 - smoothPointerY * 12;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const ribbonGradient = ctx.createLinearGradient(w * 0.12, ribbonY, w * 0.88, ribbonY + 36);
    ribbonGradient.addColorStop(0, "rgba(255,255,255,0)");
    ribbonGradient.addColorStop(0.34, "rgba(220,244,255,0.48)");
    ribbonGradient.addColorStop(0.57, "rgba(18,212,255,0.34)");
    ribbonGradient.addColorStop(0.74, "rgba(37,227,180,0.3)");
    ribbonGradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.strokeStyle = ribbonGradient;
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    for (let i = 0; i <= 150; i += 1) {
      const t = i / 150;
      const x = mix(w * 0.16, w * 0.84, t);
      const y =
        ribbonY +
        Math.sin(t * Math.PI * 1.7 + time * 0.38) * (18 + pointerSpeed * 6) +
        Math.cos(t * Math.PI * 3.2 - time * 0.24) * 6;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();

    if (!prefersReducedMotion) {
      rafId = window.requestAnimationFrame(drawMesh);
    }
  };

  resize();
  drawMesh(0);

  window.addEventListener("resize", resize);
  window.addEventListener(
    "scroll",
    () => {
      scrollY = window.scrollY || 0;
    },
    { passive: true }
  );
  window.addEventListener(
    "pointermove",
    (event) => {
      const nextX = (event.clientX / w - 0.5) * 2;
      const nextY = (event.clientY / h - 0.5) * 2;
      pointerVX = nextX - pointerX;
      pointerVY = nextY - pointerY;
      pointerX = nextX;
      pointerY = nextY;
    },
    { passive: true }
  );
  window.addEventListener(
    "pointerleave",
    () => {
      pointerX *= 0.9;
      pointerY *= 0.9;
    },
    { passive: true }
  );
  window.addEventListener("beforeunload", () => {
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
    }
  });
  }

  const revealEls = document.querySelectorAll(".reveal");
  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  } else {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.16 }
    );

    revealEls.forEach((el) => observer.observe(el));
  }

  const supportsFineHover = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  if (!prefersReducedMotion && supportsFineHover) {
    const interactiveCards = document.querySelectorAll(".glass-card");

    interactiveCards.forEach((card) => {
      const resetCardHover = () => {
        card.classList.remove("is-hovered");
        card.style.transform = "";
        card.style.setProperty("--hover-alpha", "0");
      };

      card.addEventListener("pointerenter", () => {
        card.classList.add("is-hovered");
      });

      card.addEventListener("pointermove", (event) => {
        const rect = card.getBoundingClientRect();
        const x = clamp(event.clientX - rect.left, 0, rect.width);
        const y = clamp(event.clientY - rect.top, 0, rect.height);
        const nx = rect.width > 0 ? x / rect.width : 0.5;
        const ny = rect.height > 0 ? y / rect.height : 0.5;
        const tiltX = (0.5 - ny) * 6.8;
        const tiltY = (nx - 0.5) * 7.5;

        card.style.transform = `perspective(980px) rotateX(${tiltX.toFixed(
          2
        )}deg) rotateY(${tiltY.toFixed(2)}deg) translateY(-2px)`;
        card.style.setProperty("--mx", `${(nx * 100).toFixed(1)}%`);
        card.style.setProperty("--my", `${(ny * 100).toFixed(1)}%`);
        card.style.setProperty("--hover-alpha", "1");
      });

      card.addEventListener("pointerleave", resetCardHover);
      card.addEventListener("pointercancel", resetCardHover);
    });
  }

  const terminalBody = document.getElementById("terminal-body");
  const processTimeline = document.getElementById("process-timeline");
  let terminalTimer = null;

  const clearTerminalTimer = () => {
    if (terminalTimer !== null) {
      window.clearTimeout(terminalTimer);
      terminalTimer = null;
    }
  };

  if (terminalBody) {
    const timelineItems = processTimeline
      ? Array.from(processTimeline.querySelectorAll(".timeline-item"))
      : [];
    let activeTimelineStep = 1;
    const stepCenterY = (item) => {
      if (!item) {
        return 24;
      }
      const heading = item.querySelector(".timeline-head");
      if (!heading) {
        return item.offsetTop + 24;
      }
      return item.offsetTop + heading.offsetTop + heading.offsetHeight * 0.52;
    };

    const setActiveStep = (step) => {
      if (!timelineItems.length || !step) {
        return;
      }

      if (processTimeline) {
        processTimeline.classList.add("has-active");
      }

      let activeItem = null;
      timelineItems.forEach((item) => {
        const isActive = Number(item.dataset.step) === step;
        item.classList.toggle("active", isActive);
        if (isActive) {
          activeItem = item;
        }
      });

      if (processTimeline && activeItem) {
        const focusY = stepCenterY(activeItem);
        processTimeline.style.setProperty("--focus-y", `${focusY}px`);
      }

      activeTimelineStep = step;
    };

    if (processTimeline && timelineItems.length) {
      const baseY = stepCenterY(timelineItems[0]);
      processTimeline.style.setProperty("--focus-base", `${baseY}px`);
      processTimeline.style.setProperty("--focus-y", `${baseY}px`);

      window.addEventListener("resize", () => {
        const firstStep = timelineItems[0];
        if (firstStep) {
          processTimeline.style.setProperty("--focus-base", `${stepCenterY(firstStep)}px`);
        }
        if (activeTimelineStep) {
          setActiveStep(activeTimelineStep);
        }
      });
    }

    const terminalFeed = [
      { step: 1, kind: "cmd", text: "$ ragctl ingest --source ./docs --chunk 700 --overlap 90" },
      {
        step: 1,
        kind: "log",
        text: "[ingest] parsed 184 files · chunks generated: 12460",
        pause: 520,
      },
      {
        step: 1,
        kind: "log",
        text: "[index] pgvector upsert complete · dim: 1536 · namespace: prod",
      },
      {
        step: 2,
        kind: "cmd",
        text: "$ ragctl eval --suite support_qa_v3 --model gpt-4.1-mini --trace on",
      },
      {
        step: 2,
        kind: "ok",
        text: "[eval] groundedness 0.92 · answer_f1 0.88 · hallucination 0.03",
      },
      { step: 3, kind: "cmd", text: "$ ragctl deploy --env production --canary 15%" },
      {
        step: 3,
        kind: "ok",
        text: "[deploy] health-check pass · p95 latency 1.24s · rollback ready",
      },
      {
        step: 3,
        kind: "meta",
        text: "[ready] endpoint /v1/assistant/query · tracing enabled",
        pause: 1600,
      },
    ];

    const lineLimit = 11;

    const appendLine = (entry, text = "") => {
      const line = document.createElement("div");
      line.className = `term-line ${entry.kind}`;
      line.textContent = text;
      terminalBody.appendChild(line);
      while (terminalBody.childElementCount > lineLimit) {
        terminalBody.removeChild(terminalBody.firstElementChild);
      }
      return line;
    };

    const renderStaticTerminal = () => {
      terminalBody.innerHTML = "";
      terminalFeed.forEach((entry) => appendLine(entry, entry.text));
    };

    if (prefersReducedMotion) {
      renderStaticTerminal();
      setActiveStep(3);
    } else {
      terminalBody.innerHTML = "";

      let feedIndex = 0;
      let charIndex = 0;
      let activeLine = null;

      const typeTerminal = () => {
        const entry = terminalFeed[feedIndex];

        if (!entry) {
          terminalTimer = window.setTimeout(() => {
            terminalBody.innerHTML = "";
            feedIndex = 0;
            charIndex = 0;
            activeLine = null;
            typeTerminal();
          }, 1300);
          return;
        }

        if (!activeLine) {
          setActiveStep(entry.step);
          activeLine = appendLine(entry, "");
          activeLine.classList.add("is-typing");
        }

        if (charIndex < entry.text.length) {
          activeLine.textContent += entry.text.charAt(charIndex);
          charIndex += 1;
          const speed = entry.kind === "cmd" ? 14 : 11;
          terminalTimer = window.setTimeout(typeTerminal, speed);
          return;
        }

        activeLine.classList.remove("is-typing");
        activeLine = null;
        charIndex = 0;
        feedIndex += 1;
        terminalTimer = window.setTimeout(typeTerminal, entry.pause ?? 360);
      };

      typeTerminal();
    }
  }

  const initEmailActions = () => {
    const setStatus = (el, message) => {
      if (!el) {
        return;
      }
      el.textContent = message;
      if (el.dataset.clearTimer) {
        window.clearTimeout(Number(el.dataset.clearTimer));
      }
      const timerId = window.setTimeout(() => {
        el.textContent = "";
      }, 2200);
      el.dataset.clearTimer = String(timerId);
    };

    const copyText = async (text) => {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
      }

      const field = document.createElement("textarea");
      field.value = text;
      field.setAttribute("readonly", "");
      field.style.position = "fixed";
      field.style.left = "-9999px";
      document.body.appendChild(field);
      field.select();
      document.execCommand("copy");
      document.body.removeChild(field);
    };

    const copyButtons = Array.from(document.querySelectorAll("[data-copy-email]"));

    const resolveEmailValue = (button) => {
      const explicitValue = button.getAttribute("data-copy-email-value");
      if (explicitValue) {
        return explicitValue;
      }
      return "xhani.iljard@gmail.com";
    };

    copyButtons.forEach((button) => {
      button.addEventListener("click", async () => {
        const row = button.closest(".email-copy-line");
        const statusEl = row ? row.querySelector("[data-copy-status]") : null;
        const emailValue = resolveEmailValue(button);

        try {
          await copyText(emailValue);
          setStatus(statusEl, "Email copied.");
        } catch (_error) {
          setStatus(statusEl, `Copy failed. Use ${emailValue}`);
        }
      });
    });
  };

  initEmailActions();

  const initScrollTopButton = () => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scroll-top-btn";
    button.setAttribute("aria-label", "Go to top");
    button.innerHTML = '<span aria-hidden="true">↑</span><span>Top</span>';
    document.body.appendChild(button);

    const syncButtonVisibility = () => {
      const shouldShow = (window.scrollY || 0) > 440;
      button.classList.toggle("is-visible", shouldShow);
    };

    button.addEventListener("click", () => {
      window.scrollTo({
        top: 0,
        behavior: prefersReducedMotion ? "auto" : "smooth",
      });
    });

    window.addEventListener("scroll", syncButtonVisibility, { passive: true });
    syncButtonVisibility();
  };

  initScrollTopButton();

  const initCodePreviewModal = () => {
    const openers = Array.from(document.querySelectorAll("[data-open-code-modal]"));
    if (!openers.length) {
      return;
    }

    openers.forEach((opener) => {
      const modalTarget = opener.getAttribute("data-modal-target");
      if (!modalTarget) {
        return;
      }

      const modal = document.querySelector(modalTarget);
      if (!modal) {
        return;
      }

      const closeBtn = modal.querySelector("[data-close-code-modal]");
      const codeEl = modal.querySelector("[data-code-content]");
      const snippetPath = opener.getAttribute("data-snippet-path");

      let loaded = false;

      const closeModal = () => {
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("code-modal-open");
      };

      const openModal = async () => {
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("code-modal-open");

        if (!codeEl || loaded || !snippetPath) {
          return;
        }

        try {
          const response = await fetch(snippetPath, { cache: "no-store" });
          if (!response.ok) {
            throw new Error(`Snippet fetch failed with status ${response.status}`);
          }
          const text = await response.text();
          codeEl.textContent = text;
          loaded = true;
        } catch (error) {
          codeEl.textContent = "Failed to load code preview.";
          console.error(error);
        }
      };

      opener.addEventListener("click", () => {
        openModal();
      });

      if (closeBtn) {
        closeBtn.addEventListener("click", closeModal);
      }

      modal.addEventListener("click", (event) => {
        if (event.target === modal) {
          closeModal();
        }
      });

      window.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && modal.classList.contains("is-open")) {
          closeModal();
        }
      });

      if (codeEl) {
        const block = (event) => {
          event.preventDefault();
        };

        ["copy", "cut", "contextmenu", "selectstart", "dragstart"].forEach((type) => {
          codeEl.addEventListener(type, block);
        });

        codeEl.addEventListener("keydown", (event) => {
          const key = event.key.toLowerCase();
          const isCmd = event.ctrlKey || event.metaKey;
          if (isCmd && ["a", "c", "x"].includes(key)) {
            event.preventDefault();
          }
        });
      }
    });
  };

  initCodePreviewModal();

  window.addEventListener("beforeunload", clearTerminalTimer);
})();
