document.addEventListener('DOMContentLoaded', () => {
    
    // Set time-based greeting
    const hour = new Date().getHours();
    const greetingEl = document.getElementById('greeting');
    let greeting = "Good Evening";
    
    if (hour < 12) greeting = "Good Morning";
    else if (hour < 18) greeting = "Good Afternoon";
    
    greetingEl.innerText = `${greeting}, Demo`;

    // Dummy Transaction Data
    const transactions = [
        {
            merchant: "Swiggy",
            category: "Food & Dining",
            amount: 450.00,
            type: "debit",
            date: "Today, 1:45 PM",
            icon: "restaurant-outline",
            iconClass: "icon-food"
        },
        {
            merchant: "Amazon India",
            category: "Shopping",
            amount: 2199.00,
            type: "debit",
            date: "Yesterday, 4:20 PM",
            icon: "bag-handle-outline",
            iconClass: "icon-shopping"
        },
        {
            merchant: "Salary - TechCorp",
            category: "Income",
            amount: 85000.00,
            type: "credit",
            date: "1st Oct, 9:00 AM",
            icon: "wallet-outline",
            iconClass: "icon-bills"
        },
        {
            merchant: "Netflix",
            category: "Entertainment",
            amount: 649.00,
            type: "debit",
            date: "28th Sep, 8:15 AM",
            icon: "film-outline",
            iconClass: "icon-shopping"
        }
    ];

    const txnList = document.getElementById('transaction-list');

    // Render Transactions
    transactions.forEach(txn => {
        const li = document.createElement('li');
        li.className = 'txn-item';
        
        const sign = txn.type === 'debit' ? '-' : '+';
        const colorClass = txn.type === 'debit' ? 'txn-debit' : 'txn-credit';

        li.innerHTML = `
            <div class="txn-left">
                <div class="txn-icon ${txn.iconClass}">
                    <ion-icon name="${txn.icon}"></ion-icon>
                </div>
                <div class="txn-details">
                    <h4>${txn.merchant}</h4>
                    <p>${txn.category}</p>
                </div>
            </div>
            <div class="txn-right">
                <h4 class="${colorClass}">${sign}₹${txn.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</h4>
                <p>${txn.date}</p>
            </div>
        `;
        
        txnList.appendChild(li);
    });

    // Initialize DPDP Consent Engine
    initDPDPConsent();
});

// Button Interaction
function triggerSync() {
    const btn = document.querySelector('.glass-btn');
    const icon = btn.querySelector('ion-icon');
    
    // Spin icon
    icon.style.transition = "transform 1s ease";
    icon.style.transform = "rotate(360deg)";
    btn.innerHTML = `<ion-icon name="sync-outline" style="transform: rotate(360deg);"></ion-icon> Syncing...`;
    
    setTimeout(() => {
        btn.innerHTML = `<ion-icon name="checkmark-outline"></ion-icon> Synced`;
        btn.style.background = "rgba(16, 185, 129, 0.2)";
        btn.style.borderColor = "rgba(16, 185, 129, 0.5)";
        
        setTimeout(() => {
            btn.innerHTML = `<ion-icon name="sync-outline"></ion-icon> Sync SMS`;
            btn.style.background = "";
            btn.style.borderColor = "";
        }, 2000);
    }, 1500);
}

/* ==========================================================================
   INDIA DIGITAL PERSONAL DATA PROTECTION ACT (DPDP ACT 2023) COMPLIANCE MODULE
   - Explicit Cookie Purpose & Reason Specification (Section 5)
   - Data Principal Free & Unconditional Consent (Section 6)
   - Child Protection & Age Verification (Section 9 - Minors <18 years)
   - Real-time Cookie Firing Engine & Consent Withdrawal (Section 6(4))
   ========================================================================== */

const DPDP_STORAGE_KEY = 'smartmoney_dpdp_consent_v1';

// Comprehensive registry of cookies with explicit legal reasons under DPDP Act 2023
const DPDP_COOKIE_REGISTRY = {
    necessary: {
        categoryName: "Strictly Necessary & Security",
        statutoryType: "Mandatory (DPDP Sec 4 & 7)",
        mandatory: true,
        reason: "Required to maintain authenticated financial sessions, prevent CSRF attacks, and persist your DPDP consent state.",
        cookies: [
            { name: "_sm_session_id", value: "auth_sec_" + Math.random().toString(36).substring(2, 10), maxAge: 604800, purpose: "Encrypted banking session authentication" },
            { name: "_sm_csrf_token", value: "csrf_tok_" + Math.random().toString(36).substring(2, 10), maxAge: 604800, purpose: "Cross-site request forgery prevention on SMS sync" },
            { name: "_sm_dpdp_consent", value: "dpdp_v1_granted", maxAge: 15552000, purpose: "Records Data Principal consent choices per DPDP Section 6" }
        ]
    },
    ai: {
        categoryName: "AI Forecasting & Personalization",
        statutoryType: "Optional (DPDP Sec 6)",
        mandatory: false,
        reason: "Caches personalized spending distributions and ML forecast parameters locally so 30-day predictions load instantly without re-uploading raw SMS data.",
        cookies: [
            { name: "_sm_ai_persona_cache", value: "risk_balanced_savings_high", maxAge: 7776000, purpose: "Local cache of machine-learning spending weights" },
            { name: "_sm_forecast_prefs", value: "window_30d_budget_54pct", maxAge: 7776000, purpose: "Stores user prediction horizons and expense sensitivity" }
        ]
    },
    perf: {
        categoryName: "Performance & Diagnostics",
        statutoryType: "Optional (DPDP Sec 6)",
        mandatory: false,
        reason: "Measures aggregated page load benchmarks, SMS parsing latency, and client-side error traces to optimize dashboard speed.",
        cookies: [
            { name: "_sm_perf_telemetry", value: "p95_latency_140ms_fps_60", maxAge: 5184000, purpose: "Aggregated performance and telemetry metrics" },
            { name: "_sm_sync_latency", value: "sms_sync_benchmark_ok", maxAge: 5184000, purpose: "SMS parser execution time benchmarks" }
        ]
    },
    alerts: {
        categoryName: "Financial Alerts & Nudges",
        statutoryType: "Optional (DPDP Sec 6)",
        mandatory: false,
        reason: "Tracks which high-spend warnings and investment suggestions you have already viewed to prevent repetitive nudges.",
        cookies: [
            { name: "_sm_alert_nudges", value: "entertainment_overspend_seen", maxAge: 2592000, purpose: "Suppresses duplicate high-spend advisory cards" },
            { name: "_sm_insights_viewed", value: "index_fund_recommendation_read", maxAge: 2592000, purpose: "Remembers dismissed investment suggestions" }
        ]
    }
};

// Current Session State
let dpdpState = {
    hasConsented: false,
    ageCategory: null, // 'adult' | 'minor'
    categories: {
        necessary: true,
        ai: false,
        perf: false,
        alerts: false
    },
    guardianInfo: null,
    consentTimestamp: null
};

// Initialize DPDP Consent on load
function initDPDPConsent() {
    const saved = localStorage.getItem(DPDP_STORAGE_KEY);
    
    if (saved) {
        try {
            dpdpState = JSON.parse(saved);
            applyCookiePolicies(dpdpState.categories);
            updateDPDPBadges();
        } catch (e) {
            console.error("Failed to parse saved DPDP consent:", e);
            showConsentModal(1);
        }
    } else {
        // First visit: Mandatory DPDP Age Verification & Consent Gate
        showConsentModal(1);
    }
}

// Show Modal Overlay at specified step
function showConsentModal(stepNumber) {
    const modal = document.getElementById('dpdp-modal-overlay');
    if (!modal) return;
    
    modal.style.display = 'flex';
    goToStep(stepNumber || 1);

    // Show close button only if user has previously completed consent
    const closeBtn = document.getElementById('dpdp-modal-close-btn');
    if (closeBtn) {
        closeBtn.style.display = dpdpState.hasConsented ? 'flex' : 'none';
    }
}

// Hide Modal Overlay
function closeDPDPModal() {
    const modal = document.getElementById('dpdp-modal-overlay');
    if (modal) modal.style.display = 'none';
}

// Open settings from Nav or Floating Badge
function openDPDPSettings() {
    // If previously marked as minor, go to minor step; if adult, go to adult step; otherwise step 1
    if (dpdpState.ageCategory === 'minor') {
        showConsentModal(2, 'minor');
    } else if (dpdpState.ageCategory === 'adult') {
        showConsentModal(2, 'adult');
    } else {
        showConsentModal(1);
    }
}

// Step Transition Controller
function goToStep(stepNum, subType) {
    const stepAge = document.getElementById('dpdp-step-age');
    const stepAdult = document.getElementById('dpdp-step-adult');
    const stepMinor = document.getElementById('dpdp-step-minor');
    
    const stepNav1 = document.getElementById('step-nav-1');
    const stepNav2 = document.getElementById('step-nav-2');

    // Reset panel active states
    [stepAge, stepAdult, stepMinor].forEach(el => el && el.classList.remove('active'));

    if (stepNum === 1) {
        if (stepAge) stepAge.classList.add('active');
        if (stepNav1) {
            stepNav1.classList.add('active');
            stepNav1.classList.remove('completed');
        }
        if (stepNav2) stepNav2.classList.remove('active', 'completed');
    } else if (stepNum === 2) {
        if (stepNav1) stepNav1.classList.add('completed');
        if (stepNav2) stepNav2.classList.add('active');

        const type = subType || (dpdpState.ageCategory === 'minor' ? 'minor' : 'adult');

        if (type === 'minor') {
            if (stepMinor) stepMinor.classList.add('active');
        } else {
            if (stepAdult) stepAdult.classList.add('active');
            syncTogglesWithState();
        }
    }
}

// Return to Step 1
function backToAgeStep() {
    goToStep(1);
}

// Handle Age Selection (Step 1)
function selectAgeOption(ageGroup) {
    dpdpState.ageCategory = ageGroup;

    // Visual selection indicator
    const adultCard = document.getElementById('card-age-adult');
    const minorCard = document.getElementById('card-age-minor');
    if (adultCard) adultCard.classList.toggle('selected', ageGroup === 'adult');
    if (minorCard) minorCard.classList.toggle('selected', ageGroup === 'minor');

    // Slight delay for smooth visual feedback
    setTimeout(() => {
        if (ageGroup === 'adult') {
            goToStep(2, 'adult');
        } else {
            goToStep(2, 'minor');
        }
    }, 180);
}

// Synchronize UI Checkboxes with current state
function syncTogglesWithState() {
    const aiToggle = document.getElementById('cookie-toggle-ai');
    const perfToggle = document.getElementById('cookie-toggle-perf');
    const alertsToggle = document.getElementById('cookie-toggle-alerts');

    if (aiToggle) aiToggle.checked = Boolean(dpdpState.categories.ai);
    if (perfToggle) perfToggle.checked = Boolean(dpdpState.categories.perf);
    if (alertsToggle) alertsToggle.checked = Boolean(dpdpState.categories.alerts);
}

// ==========================================
// CONSENT SUBMISSION HANDLERS
// ==========================================

// 1. Adult: Accept All
function acceptAllCookies() {
    dpdpState.hasConsented = true;
    dpdpState.ageCategory = 'adult';
    dpdpState.categories = {
        necessary: true,
        ai: true,
        perf: true,
        alerts: true
    };
    dpdpState.consentTimestamp = new Date().toISOString();

    persistAndApply("All Cookies Activated", "All cookie categories accepted with explicit DPDP purpose disclosures.");
}

// 2. Adult: Reject Non-Essential
function rejectNonEssentialCookies() {
    dpdpState.hasConsented = true;
    dpdpState.ageCategory = 'adult';
    dpdpState.categories = {
        necessary: true,
        ai: false,
        perf: false,
        alerts: false
    };
    dpdpState.consentTimestamp = new Date().toISOString();

    persistAndApply("Non-Essential Cookies Rejected", "Only strictly necessary security and session cookies will be fired.");
}

// 3. Adult: Save Custom Preferences
function saveCustomCookiePreferences() {
    const aiChecked = document.getElementById('cookie-toggle-ai')?.checked || false;
    const perfChecked = document.getElementById('cookie-toggle-perf')?.checked || false;
    const alertsChecked = document.getElementById('cookie-toggle-alerts')?.checked || false;

    dpdpState.hasConsented = true;
    dpdpState.ageCategory = 'adult';
    dpdpState.categories = {
        necessary: true,
        ai: aiChecked,
        perf: perfChecked,
        alerts: alertsChecked
    };
    dpdpState.consentTimestamp = new Date().toISOString();

    const activeCount = [true, aiChecked, perfChecked, alertsChecked].filter(Boolean).length;
    persistAndApply("Preferences Saved", `Custom preferences updated (${activeCount}/4 active categories).`);
}

// 4. Minor Protection Flow: Confirm Child-Safe Mode
function confirmMinorConsent() {
    const guardianName = document.getElementById('guardian-name')?.value.trim() || 'Parent / Guardian';
    const guardianContact = document.getElementById('guardian-contact')?.value.trim() || 'Not Provided';
    const affirmation = document.getElementById('guardian-affirmation')?.checked;

    if (!affirmation) {
        alert("Please confirm the parent or lawful guardian acknowledgment checkbox to proceed under DPDP Section 9.");
        return;
    }

    dpdpState.hasConsented = true;
    dpdpState.ageCategory = 'minor';
    dpdpState.guardianInfo = {
        name: guardianName,
        contact: guardianContact,
        verifiedAt: new Date().toISOString()
    };
    // Section 9(3): Strictly NO behavioral tracking, AI profiling, or analytics for children
    dpdpState.categories = {
        necessary: true,
        ai: false,
        perf: false,
        alerts: false
    };
    dpdpState.consentTimestamp = new Date().toISOString();

    persistAndApply("Child-Safe Mode Activated", "DPDP Section 9 protections enforced: All tracking and AI profiling cookies blocked.");
}

// 5. Withdraw All Consent (DPDP Section 6(4))
function withdrawAllConsent() {
    // Under DPDP Act Sec 6(4), ease of withdrawal must equal ease of giving consent
    dpdpState.hasConsented = false;
    dpdpState.categories = {
        necessary: true, // basic operational only
        ai: false,
        perf: false,
        alerts: false
    };
    
    // Clear all optional cookies from browser
    applyCookiePolicies(dpdpState.categories);
    localStorage.removeItem(DPDP_STORAGE_KEY);
    
    updateDPDPBadges();
    toggleCookieInspector(false);
    
    showDPDPToast("Consent Withdrawn", "All optional consent withdrawn under DPDP Section 6(4). You may set fresh preferences.");
    
    setTimeout(() => {
        showConsentModal(1);
    }, 600);
}

// Persist state, apply cookie changes, close modal, and notify
function persistAndApply(toastTitle, toastMsg) {
    localStorage.setItem(DPDP_STORAGE_KEY, JSON.stringify(dpdpState));
    applyCookiePolicies(dpdpState.categories);
    updateDPDPBadges();
    closeDPDPModal();
    showDPDPToast(toastTitle, toastMsg);
}

// ==========================================
// REAL-TIME BROWSER COOKIE ENGINE
// ==========================================

// Write or remove cookies based on active categories
function applyCookiePolicies(categories) {
    // 1. Strictly Necessary
    if (categories.necessary) {
        fireCookieCategory(DPDP_COOKIE_REGISTRY.necessary);
    }

    // 2. AI Personalization & Forecasting
    if (categories.ai) {
        fireCookieCategory(DPDP_COOKIE_REGISTRY.ai);
    } else {
        purgeCookieCategory(DPDP_COOKIE_REGISTRY.ai);
    }

    // 3. Performance & Diagnostics
    if (categories.perf) {
        fireCookieCategory(DPDP_COOKIE_REGISTRY.perf);
    } else {
        purgeCookieCategory(DPDP_COOKIE_REGISTRY.perf);
    }

    // 4. Alerts & Nudges
    if (categories.alerts) {
        fireCookieCategory(DPDP_COOKIE_REGISTRY.alerts);
    } else {
        purgeCookieCategory(DPDP_COOKIE_REGISTRY.alerts);
    }

    // Refresh audit table if open
    renderActiveCookiesTable();
}

function fireCookieCategory(categoryObj) {
    categoryObj.cookies.forEach(c => {
        // Write cookie with SameSite=Lax and proper max-age
        document.cookie = `${c.name}=${encodeURIComponent(c.value)}; max-age=${c.maxAge}; path=/; SameSite=Lax`;
    });
}

function purgeCookieCategory(categoryObj) {
    categoryObj.cookies.forEach(c => {
        // Expire cookie immediately
        document.cookie = `${c.name}=; max-age=0; path=/; SameSite=Lax`;
    });
}

// Update badges in UI (Nav and Floating button)
function updateDPDPBadges() {
    const navBadge = document.getElementById('nav-consent-status');
    const floatingTag = document.getElementById('floating-status-tag');

    if (!dpdpState.hasConsented) {
        if (navBadge) navBadge.textContent = 'DPDP Notice';
        if (floatingTag) {
            floatingTag.textContent = 'Pending';
            floatingTag.style.background = 'rgba(245, 158, 11, 0.2)';
            floatingTag.style.color = '#fbbf24';
            floatingTag.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        }
    } else if (dpdpState.ageCategory === 'minor') {
        if (navBadge) navBadge.textContent = 'Under 18 Safe';
        if (floatingTag) {
            floatingTag.textContent = 'Child Safe';
            floatingTag.style.background = 'rgba(16, 185, 129, 0.2)';
            floatingTag.style.color = '#34d399';
            floatingTag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        }
    } else {
        const activeCount = Object.values(dpdpState.categories).filter(Boolean).length;
        if (navBadge) navBadge.textContent = `18+ Adult (${activeCount}/4)`;
        if (floatingTag) {
            floatingTag.textContent = 'Compliant';
            floatingTag.style.background = 'rgba(16, 185, 129, 0.2)';
            floatingTag.style.color = '#34d399';
            floatingTag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        }
    }
}

// ==========================================
// AUDIT & COOKIE INSPECTOR DRAWER
// ==========================================

function toggleCookieInspector(forceState) {
    const drawer = document.getElementById('cookie-inspector-drawer');
    if (!drawer) return;

    if (forceState !== undefined) {
        drawer.classList.toggle('open', forceState);
    } else {
        drawer.classList.toggle('open');
    }

    if (drawer.classList.contains('open')) {
        renderActiveCookiesTable();
    }
}

// Scans document.cookie and pairs with DPDP reason registry
function renderActiveCookiesTable() {
    const tbody = document.getElementById('active-cookies-tbody');
    const countEl = document.getElementById('active-cookie-count');
    const statusEl = document.getElementById('audit-dpdp-status');
    const ageEl = document.getElementById('audit-age-category');
    const dateEl = document.getElementById('audit-consent-date');

    if (!tbody) return;

    // Parse raw document.cookie
    const rawCookies = document.cookie ? document.cookie.split(';') : [];
    const cookieMap = {};
    rawCookies.forEach(item => {
        const parts = item.split('=');
        if (parts.length >= 2) {
            cookieMap[parts[0].trim()] = parts[1].trim();
        }
    });

    tbody.innerHTML = '';
    let matchesFound = 0;

    // Cross-reference with our DPDP registry
    Object.keys(DPDP_COOKIE_REGISTRY).forEach(catKey => {
        const category = DPDP_COOKIE_REGISTRY[catKey];
        category.cookies.forEach(c => {
            if (cookieMap.hasOwnProperty(c.name)) {
                matchesFound++;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong class="code-pill">${c.name}</strong></td>
                    <td><span class="badge-tag ${category.mandatory ? 'mandatory' : 'optional'}">${category.categoryName}</span></td>
                    <td><span class="reason-summary">${c.purpose}</span></td>
                `;
                tbody.appendChild(tr);
            }
        });
    });

    if (matchesFound === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No active cookies stored in browser.</td>`;
        tbody.appendChild(tr);
    }

    if (countEl) countEl.textContent = matchesFound;
    if (statusEl) statusEl.textContent = dpdpState.hasConsented ? 'DPDP Compliant (India 2023)' : 'Consent Pending';
    if (ageEl) {
        ageEl.textContent = dpdpState.ageCategory === 'minor' ? 'Minor Data Principal (<18 Years)' : 
                            dpdpState.ageCategory === 'adult' ? 'Adult Data Principal (18+ Years)' : 'Unverified';
    }
    if (dateEl) {
        dateEl.textContent = dpdpState.consentTimestamp ? new Date(dpdpState.consentTimestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Never';
    }
}

// Toast Notification
let toastTimer = null;
function showDPDPToast(title, message) {
    const toast = document.getElementById('dpdp-toast');
    const titleEl = document.getElementById('toast-title');
    const msgEl = document.getElementById('toast-msg');

    if (!toast) return;

    if (titleEl) titleEl.textContent = title;
    if (msgEl) msgEl.textContent = message;

    toast.style.display = 'flex';

    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.style.display = 'none';
    }, 4500);
}

