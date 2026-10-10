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
