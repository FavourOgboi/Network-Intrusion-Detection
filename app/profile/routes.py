from flask import render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from . import profile
from .. import db
from werkzeug.security import generate_password_hash, check_password_hash

@profile.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_view():
    stats = {}
    # Query total predictions made by user
    result = db.session.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = :uid",
        {"uid": current_user.id}
    )
    stats['total_predictions'] = result.scalar() if result else 0

    if request.method == 'POST':
        # Change password
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        if not check_password_hash(current_user.password_hash, old_password):
            flash('Old password is incorrect.', 'danger')
        elif new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
        else:
            current_user.password_hash = generate_password_hash(new_password)
            db.session.commit()
            flash('Password updated successfully.', 'success')
            return redirect(url_for('profile.profile_view'))

    return render_template(
        'profile.html',
        user=current_user,
        stats=stats
    )
